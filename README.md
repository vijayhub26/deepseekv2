# DeepSeek-OCR-2 Pipeline

A robust, quantized OCR pipeline leveraging DeepSeek-OCR-2. It processes PDFs, handles automatic page-splitting, and corrects common OCR errors via a custom dictionary.

Optimized for 4GB VRAM GPUs (GTX 1650) via 4-bit NF4 quantization and CPU offloading.

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Important:** Always run the pipeline using the Python executable inside your virtual environment.

Run a single PDF:
```bash
.\venv\Scripts\python.exe main.py samples/batch1-0001.pdf
```

Run an entire directory:
```bash
.\venv\Scripts\python.exe main.py samples/
```

### Evaluation
Evaluation has been detached from the main pipeline. To evaluate the generated output against ground truth, run `evaluate.py` separately:

```bash
.\venv\Scripts\python.exe evaluate.py output/batch1-0001.txt ground_truth/groundtruth_b1.txt
```

## Configuration
* **`corrections.json`**: Add any recurring typos, formatting tweaks, or regex rules here. They apply automatically during the pipeline.
* **`config.json`**: High-level default pipeline configuration.
