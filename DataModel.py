
import random

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from DataProcess import load_dataset, resize_image, IMAGE_SIZE, label_num
from SavePicture import path


num = 5

class DataProcess:
    def __init__(self, path_name):

        self.train_images = None
        self.train_labels = None
        self.valid_images = None
        self.valid_labels = None
        self.test_images = None
        self.test_labels = None
        self.path_name = path_name

        self.input_shape = None

    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=num):

        images, labels = load_dataset(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)

            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels


class Model:
    def __init__(self):
        self.model = None

    def build_model(self, data_process, nb_classes=num):

        self.model = Sequential()

        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=data_process.input_shape))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, data_process, batch_size=20, nb_epoch=10, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        if not data_augmentation:
            self.model.fit(data_process.train_images,
                           data_process.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(data_process.valid_images, data_process.valid_labels),
                           shuffle=True)

        else:
            datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                         featurewise_std_normalization=False, samplewise_std_normalization=False,
                                         zca_whitening=False, rotation_range=20, width_shift_range=0.2,
                                         height_shift_range=0.2, horizontal_flip=True, vertical_flip=False)

            datagen.fit(data_process.train_images)

            self.model.fit_generator(datagen.flow(data_process.train_images, data_process.train_labels,
                                                  batch_size=batch_size),
                                     steps_per_epoch=data_process.train_images.shape[0],
                                     epochs=nb_epoch,
                                     validation_data=(data_process.valid_images, data_process.valid_labels))

    MODEL_PATH = path + '\\face.model.h5'

    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, datapro):
        score = self.model.evaluate(datapro.test_images, datapro.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # 识别人脸
    def face_detect(self, image):
        # 依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

            # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict_proba(image)
        print('result:', result)

        # 给出类别预测：0或者1
        result = self.model.predict_classes(image)

        # 返回类别预测结果
        return result[0]


if __name__ == '__main__':
    data_pro = DataProcess('C:\\Users\\13737\\Pictures\\Saved Pictures\\save_pic')
    data_pro.load()

    model = Model()
    model.build_model(data_pro)
    model.train(data_pro)
    model.save_model(file_path=(path + 'model\\face_detect.model.h5'))


