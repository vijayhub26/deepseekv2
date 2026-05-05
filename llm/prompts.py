"""
Prompt constants for DeepSeek-OCR-2 inference.

Uses <|grounding|> token for layout-preserving mode.
"""

# ===================================================================
# DeepSeek-OCR-2 Prompts
# ===================================================================

# Layout-preserving OCR prompt.
# Uses the <|grounding|> token to instruct the model to maintain
# document structure, tables, and formatting in its output.
# The prompt explicitly asks for headers/invoice numbers to reduce omissions.
LAYOUT_OCR_PROMPT = (
    "<image>\n<|grounding|>Convert the document to markdown, preserving the exact layout and spatial structure of the original document. "
    "Capture every single piece of text, including all headers, invoice numbers, "
    "company names, addresses, dates, continuation labels, and full table content. "
    "Ensure that no information is omitted and the visual hierarchy is maintained."
)

# Free OCR prompt.
# Extracts text content without structural formatting.
# Useful for simple documents where layout doesn't matter.
FREE_OCR_PROMPT = "<image>\nFree OCR."
