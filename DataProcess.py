
import os
import numpy as np
import cv2

IMAGE_SIZE = 64


def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    h, w, _ = image.shape

    longest_edge = max(h, w)

    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    BLACK = [0, 0, 0]

    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return cv2.resize(constant, (height, width))



images = []
labels = []
label_num = 0
per_dictionary = {}


def read_path(path_name):
    for dir_item in os.listdir(path_name):

        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):
            global label_num
            label_num += 1
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), -1)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)

                per_dictionary[path_name] = label_num
                images.append(image)
                labels.append(path_name)

    return images, labels



def load_dataset(path_name):
    images, labels = read_path(path_name)

    # 将输入的图片转成四维数组，尺寸为 图片数量*IMAGE_SIZE*IMAGE_SIZE*3

    images = np.array(images)
    print(images.shape)


    labels = np.array([per_dictionary[label] for label in labels])

    return images, labels



