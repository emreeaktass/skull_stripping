import albumentations as A
import numpy as np
from glob import glob
import cv2

hor_flip = A.Compose([
    A.HorizontalFlip(p=1),
])

ver_flip = A.Compose([
    A.VerticalFlip(p=1),
])

crop_pad_256 = A.Compose([
    A.CropAndPad(px=256, percent=None, keep_size=True, p=1),
])

crop_pad_128 = A.Compose([
    A.CropAndPad(px=128, percent=None, keep_size=True, p=1),
])

crop_random = A.Compose([
    A.RandomResizedCrop(width=512, height=512, p=1),
])

rotate = A.Compose([
    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=(-15, 15), p=1),
])

shift = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=(-5, 5), border_mode=cv2.BORDER_CONSTANT, p=1),
])

elastic = A.Compose([
    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, p=1),
])

grid = A.Compose([
    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=1),
])

gauss_noise = A.Compose([
    A.GaussNoise(var_limit=(10, 100), p=1),
])

gauss_blur = A.Compose([
    A.GaussianBlur(p=1),
])

brightness = A.Compose([
    A.RandomBrightnessContrast(p=1),
])

all_in_one = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.CropAndPad(px=256, percent=None, keep_size=True, p=0.4),
    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=(-15, 15), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=(-5, 5), border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.GaussNoise(var_limit=(10, 100), p=0.5),
    A.GaussianBlur(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])

if __name__ == '__main__':
    path_data = 'skull_numpy/data/'
    path_mask = 'skull_numpy/mask/'
    file_names_data = glob(path_data + '*')
    file_names_mask = glob(path_mask + '*')
    file_names_data.sort()
    file_names_mask.sort()

    for i, j in zip(file_names_data, file_names_mask):
        data = np.load(i)
        mask = np.load(j)

        transformed = all_in_one(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_all_in_one', transformed_image)
        np.save(j.split('.')[0] + '_all_in_one', transformed_mask)

        transformed = brightness(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_brightness', transformed_image)
        np.save(j.split('.')[0] + '_brightness', transformed_mask)

        transformed = gauss_blur(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_gauss_blur', transformed_image)
        np.save(j.split('.')[0] + '_gauss_blur', transformed_mask)

        transformed = gauss_noise(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_gauss_noise', transformed_image)
        np.save(j.split('.')[0] + '_gauss_noise', transformed_mask)

        transformed = grid(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_grid', transformed_image)
        np.save(j.split('.')[0] + '_grid', transformed_mask)

        transformed = elastic(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_elastic', transformed_image)
        np.save(j.split('.')[0] + '_elastic', transformed_mask)

        transformed = shift(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_shift', transformed_image)
        np.save(j.split('.')[0] + '_shift', transformed_mask)

        transformed = rotate(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_rotate', transformed_image)
        np.save(j.split('.')[0] + '_rotate', transformed_mask)

        transformed = crop_random(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_crop_random', transformed_image)
        np.save(j.split('.')[0] + '_crop_random', transformed_mask)

        transformed = crop_pad_128(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_crop_pad_128', transformed_image)
        np.save(j.split('.')[0] + '_crop_pad_128', transformed_mask)

        transformed = crop_pad_256(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_crop_pad_256', transformed_image)
        np.save(j.split('.')[0] + '_crop_pad_256', transformed_mask)

        transformed = ver_flip(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_ver_flip', transformed_image)
        np.save(j.split('.')[0] + '_ver_flip', transformed_mask)

        transformed = hor_flip(image=data, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        np.save(i.split('.')[0] + '_hor_flip', transformed_image)
        np.save(j.split('.')[0] + '_hor_flip', transformed_mask)
