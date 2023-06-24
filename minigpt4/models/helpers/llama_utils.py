import torch


def llama_decode(outputs, tokenizer, batch_size):
    """
    Decode the generated text from LLAMA model outputs.

    Args:
        outputs (torch.Tensor): The LLAMA model outputs.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for LLAMA.
        batch_size (int): The batch size of the inputs.

    Returns:
        List[List[str]]: The decoded text for each example in the batch.
    """
    generated_ids = outputs.logits.argmax(dim=-1).tolist()
    decoded_text = []
    for i in range(batch_size):
        example_ids = generated_ids[i]
        example_text = tokenizer.decode(example_ids, skip_special_tokens=True)
        decoded_text.append(example_text)
    return decoded_text


def llama_generate_text(model, input_embeds, attention_mask, tokenizer, max_length):
    """
    Generate text using the LLAMA model.

    Args:
        model (LlamaForCausalLM): The LLAMA model.
        input_embeds (torch.Tensor): The input embeddings.
        attention_mask (torch.Tensor): The attention mask.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for LLAMA.
        max_length (int): The maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    generated_ids = model.generate(
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True)
    return generated_text
