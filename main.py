# -*- coding: utf-8 -*-

"""
   Usage examples for Inception v3 on CIFAR-10 dataset
"""

from inception_v3 import *
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
import cv2
import numpy as np

### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
###


def load_data():
    (_, _), (x_train, y_train) = cifar10.load_data()
    x_train = x_train[:100]
    y_train = y_train[:100]
    print(x_train.shape)

    data_upscaled = np.zeros((100, 299, 299, 3))

    for i, img in enumerate(x_train):
        data_upscaled[i] = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('image', data_upscaled[i])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    y_train = to_categorical(y_train, 10)

    return data_upscaled, y_train


if __name__ == '__main__':
    x_train, y_train = load_data()
    print(x_train.shape)

    ### add for TensorBoard
    old_session = KTF.get_session()

    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    ###

    model = InceptionV3(num_classes=10)

    # model.summary()
    model.compile(optimizer='rmsprop',
                  loss={'predictions': 'categorical_crossentropy',
                        'aux_classifier': 'categorical_crossentropy'},
                  loss_weights={'predictions': 1., 'aux_classifier': 0.2})

    ### add for TensorBoard
    tb_cb = keras.callbacks.TensorBoard(log_dir="./log", histogram_freq=0, write_graph=True)
    cbks = [tb_cb]
    ###

    # モデルの可視化
    # plot_model(model, to_file='model.png')
    model.fit(x_train, {'predictions': y_train, 'aux_classifier': y_train},
              epochs=1, batch_size=8, callbacks=cbks, verbose=1)


    ### add for TensorBoard
    KTF.set_session(old_session)
    ###
