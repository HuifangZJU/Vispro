import matplotlib.pyplot as plt
from src.utils import get_image_var
from src.main_pipeline import get_marker_mask,get_inpainting_result,remove_background,tissue_segregation
from src.utils.image_processing_utils import save_result
from pathlib import Path

def run_Vispro(imagepath,save=False):
    img_var, img_np = get_image_var(imagepath)
    marker_mask = get_marker_mask(img_var)
    inpainted_image = get_inpainting_result(img_np,marker_mask)

    tissue_rgba,tissue_rgb,tissue_mask= remove_background(inpainted_image,tissue_threshold=100,resizing_scale=520,model_name='u2netp') # higher "tissue_threshold", less tissue region, model options are "u2net", "u2net_human_seg", "u2netp". Default is "u2net".
    segregation_image, segment_mask = tissue_segregation(tissue_rgb,tissue_mask,tissue_value=20)

    if save:
        save_result([inpainted_image,tissue_rgb,tissue_mask,segregation_image,segment_mask],imagepath[:-4])



    f,a = plt.subplots(2,3)
    a[0,0].imshow(img_np)
    a[0,1].imshow(inpainted_image)
    a[0,2].imshow(tissue_rgba)
    a[1,0].imshow(tissue_mask)
    a[1,1].imshow(segment_mask)
    a[1,2].imshow(segregation_image)
    plt.show()

test_image_path = './test_data/tissue_hires_image.png'
run_Vispro(test_image_path,save=True)

