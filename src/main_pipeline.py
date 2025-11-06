
from scipy.ndimage import label
import torch
from typing import Tuple,Optional
from skimage.color import label2rgb
from .utils.image_processing_utils import *
from .utils.model_utils import get_combined_Generator,getLamaInpainter,get_bg_model
from skimage.measure import label, regionprops
import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
import numpy as np
from .models.u2net import detect
from src.utils.image_processing_utils import overlay_mask_on_image
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)

    img = np.asarray(img)
    mask = np.asarray(mask)

    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int64)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.LANCZOS)

    return cutout


def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout


def remove(
    data,
    model,
    input_size = 320,
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_structure_size=10,
    alpha_matting_base_size=1000,
):

    # img = Image.open(io.BytesIO(data)).convert("RGB")

    # Check if data is already an ndarray
    if isinstance(data, np.ndarray):
        # Convert ndarray to RGB format if it has only 1 or 3 channels
        if data.shape[-1] == 1:  # If grayscale, convert to RGB
            img = np.repeat(data, 3, axis=-1)
        elif data.shape[-1] == 3:  # Already RGB
            img = data
        else:
            raise ValueError("Input ndarray must have either 1 or 3 channels.")
    else:
        # Assume data is binary (bytes) if not ndarray
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    img = Image.fromarray(img)
    # Convert img to a PIL Image if required for downstream processing
    mask = detect.predict(model, np.array(img),input_size).convert("L")


    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(img, mask)

    return cutout


def get_marker_mask(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Generates a binary marker mask from the input image tensor using the provided model.

    Args:
        image_tensor (torch.Tensor): The preprocessed input image tensor.
        model (torch.nn.Module): The model (generator) used to produce the marker mask.

    Returns:
        np.ndarray: A binary mask array indicating marker locations.
    """
    # Generate mask from the model output
    model = get_combined_Generator(device)
    cnn_mask_tensor = model(image_tensor)

    # Convert to NumPy and adjust dimensions
    cnn_mask = cnn_mask_tensor.cpu().detach().numpy().squeeze()
    cnn_mask = np.transpose(cnn_mask, (0, 1))

    # Binarize the mask based on a threshold
    binary_mask = binarize_array(cnn_mask, threshold=0.5)

    return binary_mask


def get_inpainting_result_batch(image_array_batch, mask_array_batch, batch_size=4):
    """
    Runs inpainting on a batch of image patches in smaller chunks to avoid out-of-memory issues.

    Args:
        image_array_batch (np.ndarray): Batch of image patches, shape (b, h, w, 3).
        mask_array_batch (np.ndarray): Batch of mask patches, shape (b, h, w).
        device (str): Device to run the inpainting model on ('cuda' or 'cpu').
        batch_size (int): Number of patches to process in each mini-batch.

    Returns:
        np.ndarray: Batch of inpainted image patches, shape (b, h, w, 3).
    """
    # Initialize inpainting model
    inpainter = getLamaInpainter(device)

    # Prepare tensors for all patches
    image_tensor = torch.from_numpy(np.transpose(image_array_batch, (0, 3, 1, 2))).float().div(255).to(device)
    mask_tensor = torch.from_numpy(mask_array_batch).unsqueeze(1).to(device)

    # Calculate total number of batches
    total_batches = (image_tensor.size(0) + batch_size - 1) // batch_size
    inpainted_results = []

    # Process in chunks of size `batch_size`
    for batch_id in range(total_batches):
        start_idx = batch_id * batch_size
        end_idx = start_idx + batch_size
        image_mini_batch = image_tensor[start_idx:end_idx]
        mask_mini_batch = mask_tensor[start_idx:end_idx]

        # Print current batch ID and total batches
        print(f"Processing batch {batch_id + 1}/{total_batches}")

        # Prepare batch dictionary
        batch = {'image': image_mini_batch, 'mask': mask_mini_batch}

        # Run inpainting on mini-batch without gradients
        with torch.no_grad():
            batch_result = inpainter(batch)

        # Convert the result back to numpy and append to results
        inpainted_mini_batch = batch_result['inpainted'].permute(0, 2, 3, 1).cpu().numpy()
        inpainted_results.append(inpainted_mini_batch)

    # Concatenate all mini-batch results along the batch dimension
    inpainted_results = np.concatenate(inpainted_results, axis=0)  # Shape: (b, h, w, 3)
    inpainted_results = np.clip(inpainted_results * 255, 0, 255).astype('uint8')

    return inpainted_results


def get_inpainting_result(image_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
    """
    Applies inpainting to the input image using a given binary mask and returns the inpainted result.

    Args:
        image_array (np.ndarray): The input image array in (H,W,C) format, normalized to [0, 1].
        mask_array (np.ndarray): The binary mask array for inpainting, in (H, W) format.

    Returns:
        np.ndarray: The inpainted image, with pixel values scaled to [0, 255].
    """
    # Convert mask and image arrays to torch tensors, and move to the device
    inpainter = getLamaInpainter(device)

    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = image_array.astype('float32') / 255
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(device)

    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device)
    # Prepare batch for inpainting model
    batch = {'image': image_tensor, 'mask': mask_tensor}

    # Run inpainting without gradients
    with torch.no_grad():
        batch = inpainter(batch)

    # Retrieve and process the inpainted result
    inpainted_image = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

    # Scale the image back to [0, 255] and convert to uint8
    inpainted_image = np.clip(inpainted_image * 255, 0, 255).astype('uint8')

    return inpainted_image


def remove_background(image_data, model_name: Optional[str] = "u2net", am=False,tissue_threshold=100, resizing_scale=320) -> Tuple:
    """
    Removes the background from an image using a specified model and saves the result.

    Args:
        src_img_path (str): The file path to the source image.
        out_img_path (str): The file path to save the output image with the background removed.
        model (str, optional): The model to use for background removal.
                               Options are "u2net", "u2net_human_seg", "u2netp". Default is "u2net".
    """
    # Perform background removal with alpha matting for smoother edges
    model = get_bg_model(model_name,device)
    tissue_image= remove(
        image_data,
        model,
        input_size = resizing_scale,
        alpha_matting=am,
        alpha_matting_foreground_threshold=200,
        alpha_matting_background_threshold=200,
        alpha_matting_erode_structure_size=25,
        alpha_matting_base_size=1000
    )


    tissue_rgba = np.asarray(tissue_image)
    tissue_mask = tissue_rgba[:, :, -1]
    tissue_rgb = get_rgb_img(tissue_rgba,tissue_threshold)
    return tissue_rgba,tissue_rgb,tissue_mask


def get_rgb_img(rgba_image,tissue_threshold) -> None:
    """
    Converts an RGBA image to an RGB image with a white background and saves it.

    Args:
        src_path (str): The file path of the source RGBA image.
        out_path (str): The file path to save the converted RGB image.
    """
    # Open the RGBA image and convert to RGBA mode to ensure an alpha channel is present

    rgba_array = np.array(rgba_image)

    # Create a binary alpha mask to standardize transparency handling
    alpha_channel = rgba_array[..., 3]
    binary_alpha = np.where(alpha_channel > tissue_threshold, 255, 0).astype(np.uint8)
    rgba_array[..., 3] = binary_alpha

    # Recreate the RGBA image with the updated alpha channel
    binary_rgba_image = Image.fromarray(rgba_array, 'RGBA')

    # Create a white background image
    white_background = Image.new('RGBA', binary_rgba_image.size, (255, 255, 255, 255))

    # Composite the RGBA image onto the white background and convert to RGB
    blended_image = Image.alpha_composite(white_background, binary_rgba_image).convert('RGB')
    blended_image = np.asarray(blended_image)
    return blended_image



def tissue_segregation(rgb_image: np.ndarray, input_mask: np.ndarray, tissue_value=200, minimum_tissue_size=200) -> Tuple:
    """
    Segregates tissue regions in a binary mask by applying thresholding
    and identifying connected components.

    Args:
        rgb_image (np.ndarray): Input RGB image used for overlaying labeled regions.
        binary_mask (np.ndarray): Input binary mask where tissue regions are indicated.

    Returns:
        np.ndarray: Labeled mask with each connected component assigned a unique integer,
                    overlaid on the original RGB image.
    """
    # Make a writable copy of the binary mask
    # Label connected components
    # Ensure binary_mask is in binary format (0 and 1)

    binary_mask = input_mask.copy()
    binary_mask[input_mask > tissue_value] = 1
    binary_mask[input_mask< tissue_value] = 0
    binary_mask = binary_mask.astype(np.uint8)
    labeled_mask, num_features = label(binary_mask, return_num=True)
    # print(f"Initial number of components: {num_features}")

    # Create a mask to store large components only
    cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Assign new labels to each component that meets the size threshold
    new_label = 1
    for region in regionprops(labeled_mask):
        if region.area >= minimum_tissue_size:
            cleaned_mask[labeled_mask == region.label] = new_label
            new_label += 1

    print(f"Number of components after removal: {np.max(cleaned_mask)}")
    overlay = overlay_mask_on_image(rgb_image, cleaned_mask, alpha=0.6)
    return overlay,cleaned_mask



