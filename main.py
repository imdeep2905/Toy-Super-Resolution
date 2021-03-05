import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from random import randint, choice

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Reshape,
    Flatten,
    Dense,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


LOG_DIR = Path("./tensorboard_logs/")


def compress(
    imgs,
    strategy=[
        cv2.INTER_LINEAR,
        cv2.INTER_NEAREST,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ],
    low=0.001,
    high=0.1,
    minmax_range=(0.001, 1.0),
):
    res = []
    scaler = MinMaxScaler(feature_range=minmax_range)
    for img in imgs:
        res.append(
            cv2.resize(
                scaler.fit_transform(img),
                dsize=(14, 14),
                interpolation=strategy[0],
            )
            + np.random.uniform(low, high, (14, 14))
        )
    return np.array(res)


def mnist_digit_pipeline():
    (y_train, _), (y_test, _) = tf.keras.datasets.mnist.load_data()
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    x_train = compress(y_train)
    x_test = compress(y_test)
    x_train = x_train.reshape(x_train.shape[0], 14, 14, 1)
    x_test = x_test.reshape(x_test.shape[0], 14, 14, 1)
    return x_train, y_train, x_test, y_test


def mnist_fashion_pipeline():
    (y_train, _), (y_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    x_train = compress(y_train)
    x_test = compress(y_test)
    x_train = x_train.reshape(x_train.shape[0], 14, 14, 1)
    x_test = x_test.reshape(x_test.shape[0], 14, 14, 1)
    return x_train, y_train, x_test, y_test


def input_pipeline(digit=True, fashion=False):
    dx_train, dy_train, dx_test, dy_test = None, None, None, None
    fx_train, fy_train, fx_test, fy_test = None, None, None, None
    if digit:
        dx_train, dy_train, dx_test, dy_test = mnist_digit_pipeline()
    if fashion:
        fx_train, fy_train, fx_test, fy_test = mnist_fashion_pipeline()
    if digit and fashion:
        return (
            np.concatenate((dx_train, fx_train)),
            np.concatenate((dy_train, fy_train)),
            np.concatenate((dx_test, fx_test)),
            np.concatenate((dy_test, fy_test)),
        )
    else:
        if digit:
            return dx_train, dy_train, dx_test, dy_test
        else:
            return fx_train, fy_train, fx_test, fy_test


class Model:
    def __init__(self, optimizer, loss_fn):
        # ################# THIS IS ONLY A DUMMY MODEL !!! ###########
        self.model = tf.keras.models.Sequential(
            [
                Conv2D(
                    32,
                    (2, 2),
                    padding="same",
                    activation="relu",
                    input_shape=[14, 14, 1],
                ),
                Flatten(),
                Dense(784, activation="relu"),
                Reshape((28, 28)),
            ]
        )
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mse", "mae"])
        # ##############################################################
        print(self.model.summary())

    def fit(self, X, Y, epochs, validation_split):
        ts_board = TensorBoard(log_dir=LOG_DIR)

        model_save = ModelCheckpoint(
            "best_model.h5",
            monitor="loss",
            verbose=1,
            save_best_only=True,
            mode="auto",
            period=1,
        )

        self.model.fit(
            X,
            Y,
            validation_split=validation_split,
            epochs=epochs,
            callbacks=[ts_board, model_save],
        )

    def evaluate(self, X, Y):
        self.model.evaluate(X, Y)

    def visualize_random_prediction(self, X, Y):
        index = randint(0, X.shape[0])
        Model.compare_result(
            Y[index],
            X[index],
            self.model.predict(np.reshape(X[index], (1, 14, 14, 1)))[0],
        )

    @staticmethod
    def compare_result(original, compressed, recovered):
        fig, axarr = plt.subplots(1, 3)
        axarr[0].imshow(original, cmap="gray")
        axarr[1].imshow(compressed, cmap="gray")
        axarr[2].imshow(recovered, cmap="gray")
        axarr[0].set_title("Original")
        axarr[1].set_title("Compressed")
        axarr[2].set_title("Recovered")
        plt.show()


def PSNR(y_true, y_pred):
    import math
    import tensorflow.keras.backend as K

    max_pixel = 1.0
    return (
        -10.0
        * (1.0 / math.log(10))
        * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
    )


def main():
    x_train, y_train, x_test, y_test = input_pipeline(digit=True, fashion=True)

    model = Model(Adam(learning_rate=0.001), PSNR)
    model.fit(x_train, y_train, 30, 0.2)
    model.evaluate(x_test, y_test)
    for _ in range(10):
        model.visualize_random_prediction(x_test, y_test)


if __name__ == "__main__":
    main()
