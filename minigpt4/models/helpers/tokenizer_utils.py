import torch
from transformers import LlamaTokenizer


class TokenizerUtils:
    """
    Utility class for tokenization using the LlamaTokenizer.
    """

    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(
            'Vision-CAIR/vicuna', use_fast=False, use_auth_token=os.environ["API_TOKEN"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.padding_side = "right"

    def tokenize_text(self, text):
        """
        Tokenizes the given text using the LlamaTokenizer.

        Args:
            text (str or List[str]): The text to tokenize.

        Returns:
            torch.Tensor: The tokenized input_ids tensor.
        """
        if isinstance(text, str):
            text = [text]

        to_regress_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        return to_regress_tokens.input_ids

    def mask_padding_tokens(self, tensor):
        """
        Masks the padding tokens in the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with padding tokens masked.
        """
        return tensor.masked_fill(tensor == self.tokenizer.pad_token_id, -100)
