import torch


def forward_pass(model, samples):
    """
    Performs a forward pass through the model.

    Args:
        model (nn.Module): The model to forward pass through.
        samples (dict): The input samples.

    Returns:
        dict: Output of the forward pass, typically containing predictions or loss.
    """
    image = samples["image"]
    output = model(image)
    return output
