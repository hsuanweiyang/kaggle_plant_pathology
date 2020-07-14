import pandas as pd
import numpy as np
import math
import os
import cv2
from tqdm import tqdm
from sys import argv
import numpy as np

PROJECT_DIR = os.path.join('..')
IMG_DIR = os.path.join(PROJECT_DIR, 'src', 'images')


def init_grabcut_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD
    mask[2*h//5:3*h//5, 2*w//5:3*w//5] = cv2.GC_FGD
    return mask


def remove_background(image, resolution):
    h, w = image.shape[:2]
    mask = init_grabcut_mask(h, w)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    cnt, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnt) != 0:
      max_cnt = max(cnt, key=cv2.contourArea)
      x, y, w, h = cv2.boundingRect(max_cnt)
      result = image[y:y+h, x:x+w]
      result = cv2.resize(result, (resolution, resolution))
    else:
      result = cv2.resize(image, (resolution, resolution))
    #plt.imshow(image)

    #result = cv2.bitwise_and(image, image, mask = mask_binary)
    #add_contours(result, mask_binary) # optional, adds visualizations
    return result, mask_binary


def preprocessing_rmbackground_v1(resolution, train=True):
    modified_img = []
    resolution = 380
    i = 0
    error = []
    for img in tqdm(img_data):
        ori_img = img.copy()
        ori_denoise = cv2.fastNlMeansDenoising(ori_img, h=5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        denoise = cv2.fastNlMeansDenoising(gray, h=5)
        filter_status = False
        base = [denoise, blur, gray]
        for canny_base in base:
            canny = cv2.Canny(canny_base, 100, 200)
            x, y, w, h = cv2.boundingRect(canny)
            if w >= resolution // 2 and h >= resolution // 2:
                filter_status = True
                break
        if not filter_status:
            img = ori_denoise
            error.append(i)
        else:
            img = ori_denoise[y:y + h, x:x + w]
        img = cv2.resize(img, (resolution, resolution))
        modified_img.append(img)
        i += 1
    modified_img = np.stack(modified_img)
    return modified_img


def preprocessing_rmbackground_v2(resolution, train=True):
    output_dir = os.path.join(IMG_DIR, 'modified')
    os.makedirs(output_dir, exist_ok=True)
    if train:
        label_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'train.csv'))
    else:
        label_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'test.csv'))
    for img_id in tqdm(label_df['image_id'][990:1000]):
        img = cv2.imread(os.path.join(IMG_DIR, '{}.jpg'.format(img_id)))
        img = remove_background(img, resolution)
        cv2.imwrite(os.path.join(output_dir, 'modified_{}.jpg'.format(img_id)), img)
