"""
Constants used in the mini-GPT4 model.
"""

# Constants for tokenizer
TOKENIZER_SPECIAL_TOKENS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

# Constants for LLAMA model
LLAMA_MODEL_NAME = "Vision-CAIR/vicuna"

# Default values for LLAMA model configuration
LLAMA_DEFAULTS = {
    "load_in_8bit": True,
    "torch_dtype": torch.float16,
    "device_map": "auto",
}

# Default values for prompt handling
PROMPT_DEFAULTS = {
    "llama_prompt_prefix": "###Human: <Img><ImageHere></Img> ",
    "llama_prompt_path": "",
    "llama_max_text_length": 32,
    "llama_end_symbol": "\n",
}
