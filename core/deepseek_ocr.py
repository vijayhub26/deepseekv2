"""
DeepSeek-OCR-2 Engine — Model loading and inference with 4-bit quantization.

Loads deepseek-ai/DeepSeek-OCR-2 with bitsandbytes NF4 quantization and
automatic device mapping (GPU + CPU offloading) to fit within 4GB VRAM.
Uses eager attention for GTX 1650 (Turing) compatibility.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig

from llm.prompts import LAYOUT_OCR_PROMPT, FREE_OCR_PROMPT

logger = logging.getLogger(__name__)

# Model identifier on HuggingFace
MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"


class DeepSeekOCR:
    """Wrapper for DeepSeek-OCR-2 model with quantized local inference.

    Attributes:
        model: The loaded transformer model.
        tokenizer: The tokenizer for the model.
        device: Primary compute device.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        use_4bit: bool = True,
        compute_dtype: torch.dtype = torch.float16,
    ):
        """Initialize and load the model.

        Args:
            model_name: HuggingFace model identifier.
            use_4bit: Whether to use 4-bit NF4 quantization (recommended
                      for <=4GB VRAM). Set False only if you have >=16GB.
            compute_dtype: Compute dtype for quantized ops. float16 for
                          Turing GPUs, bfloat16 for Ampere+.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._use_4bit = use_4bit
        self._compute_dtype = compute_dtype

    def load(self) -> None:
        """Load the model and tokenizer into memory.

        Uses 4-bit quantization and automatic device mapping to split
        the model between GPU and CPU as needed.
        """
        logger.info(f"Loading model: {self.model_name}")
        start = time.time()

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Configure quantization
        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
            "device_map": "auto",  # Automatic GPU/CPU split
            "attn_implementation": "eager",  # GTX 1650 doesn't support flash-attn
        }

        if self._use_4bit:
            logger.info("Configuring 4-bit NF4 quantization with CPU offloading...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self._compute_dtype,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = self._compute_dtype
        else:
            logger.info("Loading in full precision (bfloat16)...")
            model_kwargs["torch_dtype"] = torch.bfloat16

        # Load config and inject missing attributes for newer transformers compatibility
        logger.info("Loading and patching model config...")
        config = AutoConfig.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        # DeepSeek-OCR-2's custom code inherits improperly, so inject missing V2 attributes
        try:
            # If the custom modeling file is loaded, we can get its base config
            # But the easiest way is just to manually set known missing attributes
            missing_attrs = {
                "pad_token_id": getattr(config, "bos_token_id", 0),
                "attention_dropout": 0.0,
                "attention_bias": False,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "moe_layer_freq": 1,
                "n_shared_experts": None,
                "n_routed_experts": None,
                "num_experts_per_tok": None,
                "first_k_dense_replace": 0,
                "routed_scaling_factor": 1.0,
                "norm_topk_prob": False,
                "scoring_func": "softmax",
                "aux_loss_alpha": 0.001,
                "seq_aux": True,
                "topk_method": "greedy",
                "n_group": None,
                "topk_group": None,
            }
            for k, v in missing_attrs.items():
                if not hasattr(config, k):
                    setattr(config, k, v)
        except Exception as e:
            logger.warning(f"Failed to patch config: {e}")

        # Load model
        logger.info("Loading model weights (this may take a few minutes on first run)...")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            config=config,
            **model_kwargs,
        )
        self.model.eval()

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

        # Log device distribution
        self._log_device_map()

    def _log_device_map(self) -> None:
        """Log how the model layers are distributed across devices."""
        if hasattr(self.model, "hf_device_map"):
            device_map = self.model.hf_device_map
            devices = set(device_map.values())
            layer_counts = {}
            for dev in devices:
                count = sum(1 for v in device_map.values() if v == dev)
                dev_name = str(dev)
                layer_counts[dev_name] = count
            logger.info(f"Device distribution: {layer_counts}")
        else:
            logger.info("Device map not available (single-device mode)")

    def ocr_image(
        self,
        image_path: str,
        preserve_layout: bool = True,
        output_path: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 768,
    ) -> str:
        """Run OCR on a single image file.

        Args:
            image_path: Path to the image file.
            preserve_layout: If True, uses grounding mode for layout
                            preservation. If False, uses free OCR mode.
            output_path: Optional directory to save model outputs.
            base_size: Base resolution for the model's tile processing.
            image_size: Crop/tile size for the model.

        Returns:
            Extracted text as a markdown string.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        prompt = LAYOUT_OCR_PROMPT if preserve_layout else FREE_OCR_PROMPT
        logger.debug(f"OCR prompt: {prompt[:50]}...")

        start = time.time()

        # Build inference kwargs
        infer_kwargs = {
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "image_file": str(image_path),
            "base_size": base_size,
            "image_size": image_size,
            "crop_mode": True,
            "save_results": output_path is not None,
            "output_path": output_path if output_path else ".",
            "eval_mode": True,
        }

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self._compute_dtype):
                result = self.model.infer(**infer_kwargs)

        elapsed = time.time() - start
        logger.debug(f"Inference completed in {elapsed:.1f}s")

        # Clean up GPU memory after each inference
        self._cleanup_gpu()

        return result

    def ocr_pil_image(
        self,
        image: Image.Image,
        page_num: int,
        temp_dir: str,
        preserve_layout: bool = True,
        base_size: int = 1024,
        image_size: int = 768,
    ) -> str:
        """Run OCR on a PIL Image by saving it to a temp file first.

        The DeepSeek-OCR-2 model.infer() expects a file path, so we
        save the PIL Image to a temporary PNG file for processing.

        Args:
            image: PIL Image object (e.g., from pdf_processor).
            page_num: Page number (used for temp file naming).
            temp_dir: Directory to save the temporary image file.
            preserve_layout: Use layout-preserving grounding mode.
            base_size: Base resolution for tile processing.
            image_size: Crop/tile size for the model.

        Returns:
            Extracted text as a markdown string.
        """
        import os
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"_temp_page_{page_num:04d}.png")

        # Save PIL Image to disk for the model
        image.save(temp_path, "PNG")
        logger.debug(f"Saved temp image: {temp_path}")

        try:
            result = self.ocr_image(
                image_path=temp_path,
                preserve_layout=preserve_layout,
                base_size=base_size,
                image_size=image_size,
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"Removed temp image: {temp_path}")
                except PermissionError:
                    logger.warning(f"Could not remove temp image (file locked): {temp_path}")

        return result

    def _cleanup_gpu(self) -> None:
        """Free GPU cache memory between inferences."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def unload(self) -> None:
        """Unload model from memory and free all GPU/CPU resources."""
        logger.info("Unloading model...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._cleanup_gpu()
        logger.info("Model unloaded.")
