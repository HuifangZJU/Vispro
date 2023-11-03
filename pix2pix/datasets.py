import sys

from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
SAVE_ROOT = '/media/huifang/data/fiducial/annotation/'

def get_augmentation_parameters():
    corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    preserved_corner = np.random.choice(corners)

    # Define cropping ranges based on the preserved corner
    crop_range = 0.2
    if preserved_corner == 'top-left':
        crop_values = ((0, 0), (0, 0), (0, crop_range), (0, crop_range))
    elif preserved_corner == 'top-right':
        crop_values = ((0, 0), (0, crop_range), (0, crop_range), (0, 0))
    elif preserved_corner == 'bottom-left':
        crop_values = ((0, crop_range), (0, 0), (0, 0), (0, crop_range))
    else:  # 'bottom-right'
        crop_values = ((0, crop_range), (0, crop_range), (0, 0), (0, 0))

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),
        # iaa.Crop(percent=crop_values),
        iaa.Affine(
            scale=(0.8, 1.0),  # random scale between 80% and 100%
            rotate=(-10, 10)  # random rotation between -25 to 25 degrees
        ),
        iaa.Multiply((0.5, 1.2)),  # change brightness
        iaa.GaussianBlur(sigma=(0, 1.0))  # apply Gaussian blur
    ], random_order=False)  # apply the augmentations in random order
    return seq



def augment_image_and_keypoints(image, keypoints):
    image = image.astype(np.uint8)
    seq = get_augmentation_parameters()
    # Convert your [x, y, r] format to Keypoint format for imgaug
    kps = [ia.Keypoint(x=p[0], y=p[1]) for p in keypoints]
    kps_obj = ia.KeypointsOnImage(kps, shape=image.shape)

    # Augment image and keypoints
    image_aug, kps_aug = seq(image=image, keypoints=kps_obj)
    scale_factor = image_aug.shape[1] / image.shape[1]  # based on width
    keypoints_aug = [(kp.x, kp.y, p[2] * scale_factor) for kp, p in zip(kps_aug.keypoints, keypoints)]
    keypoints_aug = [(int(x), int(y),int(z)) for x, y, z in keypoints_aug]

    return image_aug, keypoints_aug

def augment_image_and_mask(image, mask):
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)
    seq = get_augmentation_parameters()
    image_aug, mask_aug = seq(images=[image],segmentation_maps=[mask])
    return image_aug[0],mask_aug[0]

def augment_image_mask_and_keypoints(image, mask, keypoints):
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)
    segmap = SegmentationMapsOnImage(mask, shape=image.shape)

    seq = get_augmentation_parameters()

    kps = [ia.Keypoint(x=p[0], y=p[1]) for p in keypoints]
    kps_obj = ia.KeypointsOnImage(kps, shape=image.shape)

    # Augment image and keypoints
    image_aug, mask_aug, kps_aug = seq(image=image, segmentation_maps=segmap,keypoints=kps_obj)
    scale_factor = image_aug.shape[1] / image.shape[1]  # based on width
    keypoints_aug = [(kp.x, kp.y, p[2] * scale_factor) for kp, p in zip(kps_aug.keypoints, keypoints)]
    keypoints_aug = [(int(x), int(y),int(z)) for x, y, z in keypoints_aug]

    return image_aug, mask_aug.get_arr(), keypoints_aug

def convert_cv2_to_pil(cv2_img):
    # Check the image mode
    if len(cv2_img.shape) == 3:  # RGB image
        # Convert BGR to RGB
        cv2_image_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image_rgb)
    elif len(cv2_img.shape) == 2:  # Grayscale image
        pil_image = Image.fromarray(cv2_img, 'L')
    else:
        raise ValueError("Unsupported image format")

    return pil_image

def convert_pil_to_cv2(pil_image):
    if pil_image.mode == 'RGB':
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    elif pil_image.mode == 'L':
        cv2_image = np.array(pil_image)
    else:
        raise ValueError("Unsupported image mode: {}".format(pil_image.mode))

    return cv2_image

def get_annotated_circles(annotation_path):
    in_tissue_path = os.path.join(annotation_path, 'in_tissue')
    in_tissue_circle = [circle for circle in os.listdir(in_tissue_path) if circle.endswith('image.png')]
    in_circle_meta = [[int(u), int(v), int(r), 1] for circle in in_tissue_circle for v, u, r, _ in [circle.split('_')]]

    out_tissue_path = os.path.join(annotation_path, 'auto')
    out_tissue_circle = [circle for circle in os.listdir(out_tissue_path) if circle.endswith('image.png')]
    out_circle_meta = [[int(u), int(v), int(r), 0] for circle in out_tissue_circle for v, u, r, _ in [circle.split('_')]]

    return in_circle_meta, out_circle_meta

def annotate_patches(image_size, step, circles):
    w,h = image_size
    num_patches_w = w // step
    num_patches_h = h // step

    annotation = np.zeros((num_patches_w, num_patches_h), dtype=float)
    for i in range(num_patches_w):
        for j in range(num_patches_h):
            patch_x = i * step
            patch_y = j * step
            patch_rect = (patch_x, patch_y, step, step)

            for circle in circles:
                circle_x, circle_y, circle_radius = circle[:3]
                circle_radius = circle_radius+2
                circle_rect = (circle_y - circle_radius,circle_x - circle_radius,  2 * circle_radius, 2 * circle_radius)
                if rectangles_intersect(patch_rect, circle_rect):
                    annotation[i, j] = 1.0
                    break
    return annotation

def rectangles_intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def annotate_dots(image_size, circles):
    w,h = image_size
    annotation = np.zeros((w, h), dtype=float)


    for circle in circles:
        circle_x, circle_y, circle_radius = circle[:3]
        cv2.circle(annotation,(circle_x,circle_y),circle_radius,1,thickness=-1)
    return annotation

def get_annotation_path(imagepath):
    dataset = imagepath.split('/')[6]
    index = imagepath.split('/')[7]
    index = index.split('.')
    index = index[0]
    data_path = SAVE_ROOT + dataset + '_' + index
    return data_path



def generate_heatmap(h, w, center, sigma):
    x = torch.arange(w).reshape(1, -1).repeat(h, 1).float()
    y = torch.arange(h).reshape(-1, 1).repeat(1, w).float()

    distance_from_center = (x - center[0]) ** 2 + (y - center[1]) ** 2
    heatmap = torch.exp(-distance_from_center / (2 * sigma ** 2))

    return heatmap

def generate_heatmaps(h, w, centers, sigmas):
    heatmaps = [generate_heatmap(h, w, center, sigma) for center, sigma in zip(centers, sigmas)]
    return torch.stack(heatmaps, dim=0)


class BinaryDataset(Dataset):
    def __init__(self, transforms_=None,patch_size=32,mode='test',test_group=1,aug=True):
        self.transform = transforms.Compose(transforms_)
        self.patch_size = patch_size
        self.mode = mode
        self.aug = aug
        path ='/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        self.files=[]
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            ann_percentage = line.rstrip('\n').split(' ')[-2]
            if self.mode == 'train' and int(group) != test_group:
                if float(ann_percentage)>0.9:
                    self.files.append(line)
            if self.mode == 'test' and int(group) == test_group:
                self.files.append(line)

    def __getitem__(self, index):

        image_name = self.files[index % len(self.files)].split(' ')[0]
        image_name = image_name.rstrip('\n')
        img_a = cv2.imread(image_name)
        # show_grids(image,64)
        circles = np.load(image_name.split('.')[0]+'.npy')
        if self.mode == 'train':
            img_a, circles = augment_image_and_keypoints(img_a, circles)
        img_a = convert_cv2_to_pil(img_a)
        h,w = img_a.size
        patches= annotate_patches([w,h], self.patch_size, circles)
        img_a = self.transform(img_a)
        patches = np.array(patches,dtype=np.float32)
        return {'A': img_a, 'B':patches}

    def __len__(self):
        return len(self.files)

def find_nearest_multiple_of_32(x):
    base = 32
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)

class DotDataset(Dataset):
    def __init__(self, transforms_=None,mode='test',test_group=1,aug=True):
        self.transform = transforms.Compose(transforms_)
        self.mode =mode
        self.aug = aug
        path ='/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        self.files=[]
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            if self.mode == 'train' and int(group) != test_group:
                self.files.append(line)
            if self.mode == 'test' and int(group) == test_group:
                self.files.append(line)

    def __getitem__(self, index):

        image_name = self.files[index % len(self.files)].split(' ')[0]
        image_name = image_name.rstrip('\n')
        img_a = Image.open(image_name)
        # img_a = cv2.imread(image_name)
        # show_grids(image,64)
        annotation_path = get_annotation_path(image_name)
        circles = np.load(annotation_path+'/circles.npy')
        h,w = img_a.size
        h_new = find_nearest_multiple_of_32(h)
        w_new = find_nearest_multiple_of_32(w)
        circles = np.array(circles)
        circles[:,0] = circles[:,0]*h_new/h
        circles[:, 1] = circles[:, 1] * w_new / w
        annotation = annotate_dots([w_new,h_new],circles)
        annotation = annotation.reshape(annotation.shape[0],annotation.shape[1],1)
        img_a = img_a.resize((h_new, w_new), Image.ANTIALIAS)
        if self.aug:
            img_a = convert_pil_to_cv2(img_a)
            img_a, annotation = augment_image_and_mask(img_a,annotation)
            img_a = convert_cv2_to_pil(img_a)
        f,a = plt.subplots(1,2)
        a[0].imshow(img_a)
        a[1].imshow(annotation)
        plt.show()
        annotation = np.squeeze(annotation)
        img_a = self.transform(img_a)
        annotation = np.array(annotation,dtype=np.float32)
        return {'A': img_a, 'B':annotation}

    def __len__(self):
        return len(self.files)


class AttnDataset(Dataset):
    def __init__(self, transforms_a=None,transforms_b=None,mode='test',test_group=1,aug=True):
        self.transform_a = transforms.Compose(transforms_a)
        self.transform_b = transforms.Compose(transforms_b)
        self.mode = mode
        self.aug = aug
        test_path = '/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
        train_path = '/home/huifang/workspace/data/imagelists/st_upsample_trainable_images_final.txt'

        if self.mode == 'train':
            f = open(train_path, 'r')
        elif self.mode == 'test':
            f = open(test_path, 'r')
        else:
            assert 'mode error!'
            return
        files = f.readlines()
        f.close()
        self.files = []
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            ann_percentage = line.rstrip('\n').split(' ')[-2]
            if self.mode == 'train' and int(group) != test_group:
                self.files.append(line)
                # if float(ann_percentage)>0.95:
                #     self.files.append(line)
            if self.mode == 'test' and int(group) == test_group:
                self.files.append(line)

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        image_name = img_path[0]
        image = cv2.imread(image_name)

        mask_name = image_name.split('.')[0] + '_mask_width2.png'
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        if self.aug:
            image, mask = augment_image_and_mask(image, mask[:,:,np.newaxis])
            mask = np.squeeze(mask, axis=-1)
        image = convert_cv2_to_pil(image)
        mask = convert_cv2_to_pil(mask)
        h, w = image.size
        h_new = find_nearest_multiple_of_32(h)
        w_new = find_nearest_multiple_of_32(w)
        image = image.resize((h_new, w_new), Image.ANTIALIAS)
        mask = mask.resize((h_new, w_new), Image.ANTIALIAS)
        # image = self.transform_a(image)
        # mask = self.transform_b(mask)
        return {'A': image, 'B': mask}

    def __len__(self):
        return len(self.files)


class AttnInparrelDataset(Dataset):
    def __init__(self, transforms_a=None,transforms_b=None,mode='test',test_group=1,aug=True):
        self.transform_a = transforms.Compose(transforms_a)
        self.transform_b = transforms.Compose(transforms_b)
        self.mode = mode
        self.aug = aug
        self.patch_size = 32
        test_path ='/home/huifang/workspace/data/imagelists/st_trainable_images_final.txt'
        train_path = '/home/huifang/workspace/data/imagelists/st_upsample_trainable_images_final.txt'
        if self.mode == 'train':
            f = open(train_path, 'r')
        elif self.mode == 'test':
            f = open(test_path, 'r')
        else:
            assert 'mode error!'
            return
        files = f.readlines()
        f.close()
        self.files=[]
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            ann_percentage = line.rstrip('\n').split(' ')[-2]
            if self.mode == 'train' and int(group) != test_group:
                self.files.append(line)
                # if float(ann_percentage)>0.95:
                #     self.files.append(line)
            if self.mode == 'test' and int(group) == test_group:
                self.files.append(line)

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')
        image_name = img_path[0]
        # Open the image using OpenCV
        image = cv2.imread(image_name)

        mask_name = image_name.split('.')[0] + '_mask_width2.png'
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        circles = np.load(image_name.split('.')[0] + '.npy')

        if self.aug:
            image,mask,circles = augment_image_mask_and_keypoints(image,mask[:,:,np.newaxis],circles)
            mask = np.squeeze(mask, axis=-1)
        image = convert_cv2_to_pil(image)
        mask = convert_cv2_to_pil(mask)
        h, w = image.size
        h_new = find_nearest_multiple_of_32(h)
        w_new = find_nearest_multiple_of_32(w)
        image = image.resize((h_new, w_new), Image.ANTIALIAS)
        mask = mask.resize((h_new, w_new), Image.ANTIALIAS)
        image = self.transform_a(image)
        mask = self.transform_b(mask)
        patches = annotate_patches([w_new, h_new], self.patch_size, circles)
        patches = np.array(patches, dtype=np.float32)
        return {'A': image, 'B': mask,'C':patches}

    def __len__(self):
        return len(self.files)


class CircleTrainDataset(Dataset):
    def __init__(self, path, transforms_a=None,transforms_b=None,test_group=1):
        self.transform_a = transforms.Compose(transforms_a)
        self.transform_b = transforms.Compose(transforms_b)
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        self.files = []
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            if int(group) != test_group:
                self.files.append(line)
        self.crop_size = 16
        self.max_offset= 4

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        image_name = img_path[0]
        mask_name = image_name.split('.')[0] + '_mask_width2.png'
        # Open the image using OpenCV
        image = cv2.imread(image_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        xc = int(img_path[1])
        yc = int(img_path[2])
        # Add random offsets within the specified bounds
        x_offset = random.randint(-self.max_offset, self.max_offset)
        y_offset = random.randint(-self.max_offset, self.max_offset)

        # Apply offsets while ensuring the crop remains within the image boundaries
        x_center = max(0, min(image.shape[1] - 1, xc + x_offset))
        y_center = max(0, min(image.shape[0] - 1, yc + y_offset))

        # Calculate crop coordinates
        left = max(0, x_center - self.crop_size)
        upper = max(0, y_center - self.crop_size)
        right = min(image.shape[1], x_center + self.crop_size)
        lower = min(image.shape[0], y_center + self.crop_size)

        # Crop the image using OpenCV

        cropped_image = image[upper:lower, left:right, :]
        cropped_mask = mask[upper:lower, left:right,np.newaxis]

        # cropped_image, cropped_mask = augment_image_and_mask(cropped_image, cropped_mask)
        cropped_mask = np.squeeze(cropped_mask, axis=-1)

        cropped_image = convert_cv2_to_pil(cropped_image)
        cropped_mask = convert_cv2_to_pil(cropped_mask)

        cropped_image = self.transform_a(cropped_image)
        cropped_mask = self.transform_b(cropped_mask)

        return {'A': cropped_image, 'B':cropped_mask}

    def __len__(self):
        return len(self.files)



class CircleTestDataset(Dataset):
    def __init__(self, path, transforms_=None,test_group=1):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        self.files = []
        for line in files:
            group = line.rstrip('\n').split(' ')[-1]
            if int(group) == test_group:
                self.files.append(line)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_a = Image.open(img_path[0])
        img_a = self.transform(img_a)
        return {'A': img_a}

    def __len__(self):
        return len(self.files)