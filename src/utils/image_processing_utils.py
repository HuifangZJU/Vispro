import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple
import torch

__all__ = ["binarize_array", "find_nearest_multiple_of_32", "get_image_var", "get_image_var"]

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'



def binarize_array(array, threshold):
    """
    Binarizes a numpy array based on a threshold determined by the given percentile.

    :param array: numpy array to be binarized
    :param percentile: percentile value used to determine the threshold, defaults to 50 (median)
    :return: binarized numpy array
    """
    binary_array = (array >= threshold).astype(int)

    return binary_array

def find_nearest_multiple_of_32(x):
    base = 32
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)

def get_image_var(image_name: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads an image, resizes it to the nearest multiple of 32,
    and transforms it to a tensor format compatible with a model.

    Args:
        image_name (str): The path to the image file.

    Returns:
        Tuple[torch.Tensor, np.ndarray]: A tuple containing:
            - img_var: The transformed image tensor ready for model input.
            - img_np: The image as a NumPy array in its resized form.
    """
    # Load the image and extract dimensions
    img_pil = Image.open(image_name.split(' ')[0])
    h, w = img_pil.size

    # Resize dimensions to nearest multiple of 32
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new))

    # Convert to NumPy array and prepare tensor
    img_np = np.array(img_pil)
    transforms_rgb = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_var = transforms_rgb(img_pil)
    img_var = torch.unsqueeze(img_var, dim=0).to(device)

    return img_var, img_np

def save_result(images, filename_prefix, suffix=".png"):
    for i, img_array in enumerate(images):
        img_array = np.array(img_array)

        if img_array.ndim == 2:
            unique_vals = np.unique(img_array)
            if np.array_equal(unique_vals, [0, 1]):
                # Binary mask
                img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
            elif np.issubdtype(img_array.dtype, np.integer):
                # Label mask with values like 0,1,2,...
                max_val = img_array.max()
                if max_val > 0:
                    img_array = (img_array / max_val * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
            else:
                img_array = img_array.astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
        elif img_array.ndim == 3:
            if img_array.shape[2] == 3:
                img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
            elif img_array.shape[2] == 4:
                img = Image.fromarray(img_array.astype(np.uint8), mode='RGBA')
            else:
                raise ValueError(f"Unsupported channel size: {img_array.shape[2]}")
        else:
            raise ValueError(f"Unsupported image shape: {img_array.shape}")

        output_filename = f"{filename_prefix}_{i}{suffix}"
        img.save(output_filename)


def overlay_mask_on_image(rgb_image: np.ndarray, label_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a label mask on an RGB image using alpha blending.

    Args:
        rgb_image (np.ndarray): Original RGB image, shape (H, W, 3), dtype uint8.
        label_mask (np.ndarray): Integer label mask, shape (H, W), values 0 = background.
        alpha (float): Transparency factor for overlay.

    Returns:
        np.ndarray: Blended RGB image.
    """
    # Convert image to float [0, 1]
    img = rgb_image.astype(np.float32) / 255.0

    # Generate random colors for each label (0 is background, so skip it)
    labels = np.unique(label_mask)
    labels = labels[labels != 0]  # exclude background
    color_map = {label: np.random.rand(3) for label in labels}  # random RGB [0,1]

    # Prepare overlay mask
    overlay = np.zeros_like(img)

    for label, color in color_map.items():
        mask = label_mask == label
        for c in range(3):  # R, G, B
            overlay[..., c][mask] = color[c]

    # Alpha blend overlay with original image
    blended = (1 - alpha) * img + alpha * overlay
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return blended