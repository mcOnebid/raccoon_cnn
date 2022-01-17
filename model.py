import numpy as np
import time
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from data_processing import DataProcessor

import datetime


class Model:
    INCEPTION_V3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
    MOBILENET_V2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    RESNET50_V2 = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
    RESNET152_V2 = "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5"

    MODELS_DICT = {  # slownik dla modeli
        0: MOBILENET_V2,
        1: INCEPTION_V3,
        2: RESNET50_V2,
        3: RESNET152_V2
    }

    MODELS_SIZE = {  # wielkosci wybranych modeli #TODO Testowanie innych wielkosci (ratio musi byc zachowane!)
        0: (224, 224, 3),
        1: (299, 299, 3),
        2: (224, 224, 3),
        3: (224, 224, 3)
    }

    VALUE_RANGE = [0, 1]  # dla wszystkich wybranych modeli wartosci pikseli musza byc standaryzowane [0,255] -> [0,1]

    def __init__(self, model_number, data_dir, batch_size, split_ratio, epoch = 100):
        self.data_precessor = DataProcessor(data_dir,img_height=self.MODELS_SIZE[model_number][0],
                                       img_width= self.MODELS_SIZE[model_number][1], batch_size= batch_size,
                                       split_ratio= split_ratio)
        self.data_precessor.buffered_prefetching() #pre-fetching
        feature_extractor_model = self.MODELS_DICT[model_number]
        feature_extractor_layer = hub.KerasLayer(
            feature_extractor_model,
            input_shape = self.MODELS_SIZE[model_number],
            trainable= False #modyfikowane sa tylko nowe warstwy
        )

        self.model = tf.keras.Sequential([
                  feature_extractor_layer,
                  tf.keras.layers.Dense(len(self.data_precessor.class_names))
                ])

        self.model.summary()
        self.epoch_number= epoch


    def train(self):
        self.model.compile( #kofiguracja treningu
            optimizer=tf.keras.optimizers.Adam(), #Tu mozna porzezbic oczywiscie z optymalizacja gradientowa ADAM standard
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc']
        )

        #LOGI
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1)

        history = self.model.fit(self.data_precessor.train_ds, ### NO I OGIEN NA TŁOKI !
                            validation_data=self.data_precessor.validation_ds,
                            epochs=self.epoch_number,
                            callbacks=tensorboard_callback)

        t = time.time()

        export_path = "/saved_models/{}".format(int(t))
        self.model.save(export_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_dir",
                        help="Sciezka do datasetu",
                        type=str)
    parser.add_argument("-m",
                        "--model_number",
                        help="Numer modelu - 0: moblinetv2, 1:inception v3, 2: resnet50v2, 3:resnet152v2", type=int)
    parser.add_argument("-b",
                        "--batch_size",
                        help="Rozmiar paczki (mini-batchu) jesli brakuje pamieci sprobuj mniejsza wartosc",
                        type=int, default=8, required= False)
    parser.add_argument("-s",
                        "--split_ratio",
                        help="Ile procent zbioru na walidacje",
                        type=float, default=0.2, required= False)


    args = parser.parse_args()

    model = Model(args.model_number,data_dir=args.data_dir,batch_size=args.batch_size,split_ratio=args.split_ratio)
    model.train()
