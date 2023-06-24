import random


def load_prompts(prompt_path, prompt_template):
    """
    Load prompts from a file and format them using the given template.

    Args:
        prompt_path (str): Path to the prompt file.
        prompt_template (str): Template string for formatting prompts.

    Returns:
        list: List of formatted prompts.
    """
    with open(prompt_path, 'r') as f:
        raw_prompts = f.read().splitlines()

    filtered_prompts = [
        raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
    formatted_prompts = [prompt_template.format(p) for p in filtered_prompts]

    return formatted_prompts


def get_random_prompt(prompt_list):
    """
    Get a random prompt from the given list.

    Args:
        prompt_list (list): List of prompts.

    Returns:
        str: Randomly selected prompt.
    """
    return random.choice(prompt_list)
