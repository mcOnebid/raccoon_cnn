import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class DataProcessor:
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, data_dir, img_height=640, img_width=640, batch_size=32, split_ratio=0.2):
        '''
        Zainicjowanie datasetu treningowego,testowego i walidacyjnego
        :param data_dir: sciezka do datasetu    |-data_dir
                                                    |-class1
                                                    |-class2
                                                    ...
                                                    |-class_n
        :param img_height: wysokosc docelowa obrazu -> zgodna z modelem sieci
        :param img_width: szerokosc
        :param batch_size: wielkosc batchu ('mini-batchu), jeśli malo pamieci -> zmniejszamy (np.1,2,4,8)
        :param split_ratio: stosunek podzialu na zbior testowy i trenujacy

        Atrybuty DataProcessor:
        train_ds -> zbior trenujacy
        test_ds -> zbior testowy
        validation_ds -> zbior walidacyjny
        '''
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=split_ratio,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        class_names = self.train_ds.class_names  # standaryzacja danych

        self.validation_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=split_ratio,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        self.class_names = np.array(self.train_ds.class_names)
        self.save_number_to_file()
        print('Klasa obiektu:', *class_names, sep='\n')

    def save_number_to_file(self):
        '''
        Zapis ilosci zdjec do pliku zeby mozna bylo oceniac kiedy dotrenowywac model
        :return:
        '''
        with open('number_of_images.txt', 'a') as f:
            f.write(str(len(self.train_ds.file_paths) + len(self.validation_ds.file_paths)))
            f.close()

    def normalize_data(self):
        '''
        Standaryzacja danych: RGB [0,255] -> [0,1]
        '''
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        self.validation_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y))

    def buffered_prefetching(self):
        '''
        Prefetching - zaciagnie danych (gdy trening jest na kroku x zaciagne sa dane dla x+1)
        :return:
        '''
        self.train_ds = self.train_ds.prefetch(buffer_size=self.AUTOTUNE)
        self.validation_ds = self.validation_ds.prefetch(buffer_size=self.AUTOTUNE)

