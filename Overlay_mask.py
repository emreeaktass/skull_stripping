from glob import glob
import pydicom as dcm
import cv2
from scipy import ndimage
import numpy as np
import tensorflow as tf
import albumentations as A
import os
import warnings
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    if rescale:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return img


def get_first_of_dicom_field_as_int(x):
    if type(x) == dcm.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


if __name__ == '__main__':
    saving_inme_yok_path = '../work/dataset/INME_YOK/'
    saving_iskemi_path = '../work/dataset/ISKEMI/'
    saving_kanama_path = '../work/dataset/KANAMA/'
    path_data_inme_yok = '../datasett/TRAINING/INMEYOK/DICOM/*'
    path_data_iskemi = '../datasett/TRAINING/ISKEMI/DICOM/*'
    path_mask_iskemi = '../datasett/TRAINING/ISKEMI/MASK/*'
    path_data_kanama = '../datasett/TRAINING/KANAMA/DICOM/*'
    path_mask_kanama = '../datasett/TRAINING/KANAMA/MASK/*'
    file_names_inme_yok = glob(path_data_inme_yok)
    file_names_iskemi = glob(path_data_iskemi)
    file_mask_iskemi = glob(path_mask_iskemi)
    file_names_kanama = glob(path_data_kanama)
    file_mask_kanama = glob(path_mask_kanama)
    file_names_inme_yok.sort()
    file_names_iskemi.sort()
    file_names_kanama.sort()
    file_mask_iskemi.sort()
    file_mask_kanama.sort()

    model = tf.keras.models.load_model('record8/', compile=False)

    resize = A.Compose([
        A.Resize(width=256, height=256, p=1),
    ])

    # for i in file_names_inme_yok:
    #     print(i)
    #     ds = dcm.dcmread(i)
    #     a = ds.pixel_array
    #     window_center, window_width, intercept, slope = get_windowing(ds)
    #     output = window_image(a, window_center, window_width, intercept, slope, rescale=True)
    #     output = output.astype(np.uint8)
    #     transformed = resize(image=output)
    #     transformed_image = transformed["image"]
    #     transformed_image = transformed_image.reshape(1, 256, 256, 1)
    #
    #     y_pred = model.predict(transformed_image)
    #     xx = y_pred[0]
    #     xx[xx >= 0.5] = 1
    #     xx[xx < 0.5] = 0
    #
    #     xx = ndimage.binary_fill_holes(xx).astype(np.uint8)
    #     contours, hierarchy = cv2.findContours(xx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     for aa in range(len(contours)):
    #         cv2.drawContours(xx, [contours[aa]], -1, (1), cv2.FILLED)
    #
    #     res = (xx * transformed_image[0])
    #     # cv2.imshow('trans', transformed_image[0, :, :, 0])
    #     # cv2.imshow('res', res)
    #     # cv2.imshow('pred', xx[:,:,0] * 255)
    #
    #
    #     # cv2.imshow('emer', np.hstack((transformed_image[0, :, :, 0], res[:,:,0] * 255, xx[:,:,0] * 255)))
    #     # cv2.imshow('predicted', np.hstack((transformed_image[0, :, :, 0] * 255, res * 255, xx[:,:,0] * 255)))
    #     # time.sleep(1.5)
    #     # cv2.waitKey(1)
    #     mm = np.zeros((256, 256, 2), dtype=np.uint8)
    #     ll = np.array(0).astype(np.uint8)
    #
    #     # np.save(saving_inme_yok_path + 'data/' + i.split('.')[2].split('/')[-1], res)
    #     # np.save(saving_inme_yok_path + 'comb/' + i.split('.')[2].split('/')[-1], mm)
    #     # np.save(saving_inme_yok_path + 'label/' + i.split('.')[2].split('/')[-1], ll)
    #
    # for i in range(len(file_names_iskemi)):
    #     print(i)
    #     ds = dcm.dcmread(file_names_iskemi[i])
    #     a = ds.pixel_array
    #     mask = cv2.imread(file_mask_iskemi[i])
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     mask[mask>0] = 1
    #     window_center, window_width, intercept, slope = get_windowing(ds)
    #     output = window_image(a, window_center, window_width, intercept, slope, rescale=True)
    #     output = output.astype(np.uint8)
    #     transformed = resize(image=output, mask=mask)
    #     transformed_image = transformed["image"]
    #     transformed_mask = transformed["mask"]
    #     transformed_image = transformed_image.reshape(1, 256, 256, 1)
    #
    #     y_pred = model.predict(transformed_image)
    #     xx = y_pred[0]
    #     xx[xx >= 0.5] = 1
    #     xx[xx < 0.5] = 0
    #
    #     xx = ndimage.binary_fill_holes(xx).astype(np.uint8)
    #
    #     contours, hierarchy = cv2.findContours(xx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     for aa in range(len(contours)):
    #         cv2.drawContours(xx, [contours[aa]], -1, (1), cv2.FILLED)
    #
    #     res = (xx * transformed_image[0])
    #     # cv2.imshow('trans', transformed_image[0, :, :, 0])
    #     # cv2.imshow('res', res)
    #     # cv2.imshow('pred', xx[:,:,0] * 255)
    #     # cv2.imshow('mask', transformed_mask*255)
    #     # cv2.waitKey(1)
    #     # time.sleep(1.5)
    #     mm = np.zeros((256, 256, 2), dtype=np.uint8)
    #     ll = np.array(1).astype(np.uint8)
    #     mm[:,:,0] = transformed_mask
    #     np.save(saving_iskemi_path + 'data/' + file_names_iskemi[i].split('.')[2].split('/')[-1], res)
    #     np.save(saving_iskemi_path + 'mask/' + file_names_iskemi[i].split('.')[2].split('/')[-1], transformed_mask)
    #     np.save(saving_iskemi_path + 'comb/' + file_names_iskemi[i].split('.')[2].split('/')[-1], mm)
    #     np.save(saving_iskemi_path + 'label/' + file_names_iskemi[i].split('.')[2].split('/')[-1], ll)


    for i in range(len(file_names_kanama)):
        print(i)
        ds = dcm.dcmread(file_names_kanama[i])
        a = ds.pixel_array
        mask = cv2.imread(file_mask_kanama[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[mask>0] = 1
        window_center, window_width, intercept, slope = get_windowing(ds)
        output = window_image(a, window_center, window_width, intercept, slope, rescale=True)
        output = output.astype(np.uint8)
        transformed = resize(image=output, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        transformed_image = transformed_image.reshape(1, 256, 256, 1)

        y_pred = model.predict(transformed_image)
        xx = y_pred[0]
        xx[xx >= 0.5] = 1
        xx[xx < 0.5] = 0
        xx = ndimage.binary_fill_holes(xx).astype(np.uint8)
        contours, hierarchy = cv2.findContours(xx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for aa in range(len(contours)):
            cv2.drawContours(xx, [contours[aa]], -1, (1), cv2.FILLED)

        res = (xx * transformed_image[0])
        # cv2.imshow('trans', transformed_image[0, :, :, 0])
        # cv2.imshow('res', res)
        # cv2.imshow('pred', xx[:,:,0] * 255)
        # cv2.imshow('mask', transformed_mask*255)
        # cv2.waitKey(1)
        # time.sleep(1)

        mm = np.zeros((256, 256, 2), dtype=np.uint8)
        ll = np.array(2).astype(np.uint8)
        mm[:,:,1] = transformed_mask
        np.save(saving_kanama_path + 'data/' + file_names_kanama[i].split('.')[2].split('/')[-1], res)
        np.save(saving_kanama_path + 'mask/' + file_names_kanama[i].split('.')[2].split('/')[-1], transformed_mask)
        np.save(saving_kanama_path + 'comb/' + file_names_kanama[i].split('.')[2].split('/')[-1], mm)
        np.save(saving_kanama_path + 'label/' + file_names_kanama[i].split('.')[2].split('/')[-1], ll)
