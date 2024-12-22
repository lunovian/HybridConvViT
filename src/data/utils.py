import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

def show_image(tensor, title=None):
    """
    Utility function to display a single image tensor.
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # remove batch dimension
    image = to_pil_image(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

def save_image(tensor, path):
    """
    Utility function to save a single image tensor to file.
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # remove batch dimension
    image = to_pil_image(image)
    image.save(path)

def tensor_to_numpy(tensor):
    """
    Convert a tensor to a numpy array for visualization.
    """
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)