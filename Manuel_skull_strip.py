import numpy as np
import cv2
import time
import keyboard
import pydicom as dcm
import warnings

warnings.filterwarnings("ignore")

points = []
counter = 0


def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    img = (img * slope + intercept)  # for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[img < img_min] = img_min  # set img_min for all HU levels less than minimum HU level
    img[img > img_max] = img_max  # set img_max for all HU levels higher than maximum HU level
    if rescale:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return img


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
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


def mouse_points(event, x, y, flags, params):
    global counter
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


if __name__ == '__main__':

    path = 'skull_numpy/'
    with open('file_path.txt') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    for i in lines:
        print(i)
        k = i.split('/')
        dim = (512, 512)
        ds = dcm.dcmread(i)
        a = ds.pixel_array
        window_center, window_width, intercept, slope = get_windowing(ds)
        output = window_image(a, window_center, window_width, intercept, slope, rescale=True)
        output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)
        output_not_changed = output.copy().astype(np.uint8)

        while True:
            image = output.copy().astype(np.uint8)
            mask = np.zeros(image.shape, dtype=np.float32)
            if len(points) > 0:
                for a in range(len(points)):
                    cv2.circle(image, (points[a]), 1, (0, 255, 0), cv2.FILLED)

            if len(points) > 2:
                ctr = np.array(points).astype(np.int32)
                cv2.fillPoly(mask, [ctr], 1)
                cv2.drawContours(image, [ctr], 0, [0, 255, 0], 1)

                cv2.imshow('cropped', mask)
            cv2.imshow('emre', image)

            cv2.setMouseCallback('emre', mouse_points)
            cv2.waitKey(1)

            if cv2.waitKey(1) & keyboard.is_pressed('z'):
                keyboard.release('z')
                time.sleep(0.1)
                try:
                    del points[-1]
                except:
                    print('List index out of boundation!!')

            if cv2.waitKey(2) & keyboard.is_pressed('n'):
                time.sleep(0.1)
                keyboard.release('n')
                points.clear()
                break

            if cv2.waitKey(2) & keyboard.is_pressed('s'):
                time.sleep(0.1)
                keyboard.release('s')
                print('Saving to the path {} with name {}'.format(('data/' + k[3]), (k[5].split('.')[0])))
                np.save(path + 'data/' + k[3] + '_' + k[5].split('.')[0], output)
                np.save(path + 'mask/' + k[3] + '_' + k[5].split('.')[0], mask.astype(np.uint8))
