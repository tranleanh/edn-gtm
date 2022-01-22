import os
from PIL import Image
import numpy as np
import cv2
from random import randint
from .dcp import estimate_transmission

from .networks import img_size


# img_size = 512

RESHAPE = (img_size,img_size)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', 'bmp']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    cnt=0
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        cnt+=1
        print(cnt, n_images)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }


def preprocess_cv2_image(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def preprocess_depth_img(cv_img):
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    img = np.reshape(img, (RESHAPE[0], RESHAPE[1], 1))
    img = 2*(img - 0.5)
    return img


def preprocess_image_cv2_rancrop(img_A, img_B):

    t = estimate_transmission(img_A)

    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

    h, w, _ = img_A.shape

    min_wh = np.amin([h, w])

    # crop_sizes = [1600, 1800, 2000, 2200, 2400]
    crop_sizes = [int(min_wh*0.4), int(min_wh*0.5), int(min_wh*0.6), int(min_wh*0.7), int(min_wh*0.8)]

    images_A = []
    images_B = []

    for crop_size in crop_sizes:

        x1, y1 = randint(1, w-crop_size-1), randint(1, h-crop_size-1)

        cropA = img_A[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropA = cv2.resize(cropA, (RESHAPE))
        cropA = np.array(cropA)
        cropA = (cropA - 127.5) / 127.5

        crop_t = t[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        crop_t = preprocess_depth_img(crop_t)

        cropA = np.concatenate((cropA, crop_t), axis=2)

        cropB = img_B[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropB = cv2.resize(cropB, (RESHAPE))
        cropB = np.array(cropB)
        cropB = (cropB - 127.5) / 127.5

        images_A.append(cropA)
        images_B.append(cropB)

    img_A = cv2.resize(img_A, (RESHAPE))
    img_A = np.array(img_A)
    img_A = (img_A - 127.5) / 127.5

    t = preprocess_depth_img(t)

    img_A = np.concatenate((img_A, t), axis=2)

    img_B = cv2.resize(img_B, (RESHAPE))
    img_B = np.array(img_B)
    img_B = (img_B - 127.5) / 127.5

    images_A.append(img_A)
    images_B.append(img_B)

    return images_A, images_B



def load_images_with_crop_data_aug(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    cnt=0
    for path_A, path_B in zip(all_A_paths, all_B_paths):

        img_A, img_B = cv2.imread(path_A), cv2.imread(path_B)

        processed_imgs_A, processed_imgs_B = preprocess_image_cv2_rancrop(img_A, img_B)

        for imgA in processed_imgs_A: images_A.append(imgA)
        for imgB in processed_imgs_B: images_B.append(imgB)

        images_A_paths.append(path_A)
        images_B_paths.append(path_B)

        cnt+=1
        print(cnt, n_images)
        if len(images_A_paths) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }


def preprocess_image_cv2_rancrop_flip(img_A, img_B):

    t = estimate_transmission(img_A)
    t_flip = cv2.flip(t, 1)

    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

    img_A_flip = cv2.flip(img_A, 1)
    img_B_flip = cv2.flip(img_B, 1)

    h, w, _ = img_A.shape

    min_wh = np.amin([h, w])

    # crop_sizes = [1600, 1800, 2000, 2200, 2400]
    crop_sizes = [int(min_wh*0.4), int(min_wh*0.5), int(min_wh*0.6), int(min_wh*0.7), int(min_wh*0.8)]


    images_A = []
    images_B = []

    for crop_size in crop_sizes:

        x1, y1 = randint(1, w-crop_size-1), randint(1, h-crop_size-1)

        # Original

        cropA = img_A[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropA = cv2.resize(cropA, (RESHAPE))
        cropA = np.array(cropA)
        cropA = (cropA - 127.5) / 127.5

        crop_t = t[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        crop_t = preprocess_depth_img(crop_t)

        cropA = np.concatenate((cropA, crop_t), axis=2)

        cropB = img_B[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropB = cv2.resize(cropB, (RESHAPE))
        cropB = np.array(cropB)
        cropB = (cropB - 127.5) / 127.5

        images_A.append(cropA)
        images_B.append(cropB)


        # Horizontal Flip

        cropA = img_A_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropA = cv2.resize(cropA, (RESHAPE))
        cropA = np.array(cropA)
        cropA = (cropA - 127.5) / 127.5

        crop_t = t_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        crop_t = preprocess_depth_img(crop_t)

        cropA = np.concatenate((cropA, crop_t), axis=2)

        cropB = img_B_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropB = cv2.resize(cropB, (RESHAPE))
        cropB = np.array(cropB)
        cropB = (cropB - 127.5) / 127.5

        images_A.append(cropA)
        images_B.append(cropB)


    # Original

    img_A = cv2.resize(img_A, (RESHAPE))
    img_A = np.array(img_A)
    img_A = (img_A - 127.5) / 127.5

    t = preprocess_depth_img(t)

    img_A = np.concatenate((img_A, t), axis=2)

    img_B = cv2.resize(img_B, (RESHAPE))
    img_B = np.array(img_B)
    img_B = (img_B - 127.5) / 127.5

    images_A.append(img_A)
    images_B.append(img_B)


    # Horizontal Flip

    img_A = cv2.resize(img_A_flip, (RESHAPE))
    img_A = np.array(img_A)
    img_A = (img_A - 127.5) / 127.5

    t = preprocess_depth_img(t_flip)

    img_A = np.concatenate((img_A, t), axis=2)

    img_B = cv2.resize(img_B_flip, (RESHAPE))
    img_B = np.array(img_B)
    img_B = (img_B - 127.5) / 127.5

    images_A.append(img_A)
    images_B.append(img_B)

    return images_A, images_B



def load_images_with_crop_flip_data_aug(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    cnt=0
    for path_A, path_B in zip(all_A_paths, all_B_paths):

        img_A, img_B = cv2.imread(path_A), cv2.imread(path_B)

        processed_imgs_A, processed_imgs_B = preprocess_image_cv2_rancrop_flip(img_A, img_B)

        for imgA in processed_imgs_A: images_A.append(imgA)
        for imgB in processed_imgs_B: images_B.append(imgB)

        images_A_paths.append(path_A)
        images_B_paths.append(path_B)

        cnt+=1
        print(cnt, n_images)
        if len(images_A_paths) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }