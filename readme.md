# Vispro
Vispro is a command line tool for Spatial Transcriptomics image processing, including modules of fiducial marker detection, image restoration, tissue detection, and disconnected tissue segregation. It is dominantly develped and tested on high-resolution images, with resoluton of ~2000*2000. It can be applied to original microscopy image, with resolution of ~20,000 *~20,000 by dividing them into smaller patches and stitch the result togetehr. The project is built with PyTorch and Python and can run efficiently on a GPU.
### Requirements
* Python 3.8 or later
* CUDA (for GPU support, optional but recommended)
* Python Libraries:
  * numpy>=1.18.0
  * torch>=1.7.0
  * scikit-image>=0.16.2
  * Pillow>=7.0.0
  * torchvision>=0.8.0
  * scipy>=1.4.1
  * matplotlib>=3.1.3
  * pymatting>=1.1.5
  * kornia>=0.5.0

### Installation

Vispro is a pure-python based tool. We recommend use a miniconda environment for library management and running the code.

1. Clone the repository
```bash
git clone https://github.com/HuifangZJU/Vispro.git
cd Vispro
```
2. Set up environment

 * 2.1 Set up a conda environment with all dependency libraries (recommended)
```bash
conda env create -f environment.yml
conda activate Vispro
```
For the installation of miniconda, please refer to the link https://docs.anaconda.com/miniconda/.
 * 2.2 Install dependency directly with pip
```bash
pip install -r requirements.txt
```

### Usage
The full list of pretrained models can be found [Direct Download](https://drive.google.com/file/d/13M_bHQfy8crnF1Dg8PXH0NTGAYNmi78M/view?usp=drive_link)


Process the high-resolution image only.

```bash
python process_high_res.py --image_path /test_data/tissue_hires_image.png
```
Process the original large image.

```bash
python process_large_image.py --high_res_image_path /test_data/151673/spatial/tissue_hires_image.png --original_image_path /test_data/151673/151673_full_image.tif
```

Licensed under [MIT License](./LICENSE)
