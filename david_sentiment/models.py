from typing import List, Tuple

import numpy as np
from keras import layers, models, optimizers
from keras.preprocessing.sequence import pad_sequences


class BirectionalRNN:
    """Simple Birectional RNN|LSTM Model"""

    def __init__(self, vocab_size: int, ndim: int, maxlen: int):
        self.vocab_size = vocab_size
        self.ndim = ndim
        self.maxlen = maxlen

    def reverse(self, x_train: List[int], x_test: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Reverse the input sequences for the RNN layer."""
        x_train = [x[::-1] for x in x_train]
        x_test = [x[::-1] for x in x_test]
        x_train = pad_sequences(x_train, self.maxlen)
        x_test = pad_sequences(x_test, self.maxlen)
        return x_train, x_test

    def compile_model(self, lstm_output=32) -> models.Sequential:
        """RNN layer model definition."""
        model = models.Sequential(name="Bidirectional RNN|LSTM")
        model.add(layers.Embedding(self.vocab_size, self.ndim))
        model.add(layers.LSTM(lstm_output))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
        return model


class ConvolutionalNN:
    """Simple 1Dimentional Convolutional Model."""

    def __init__(self, vocab_size: int, ndim: int, maxlen: int):
        self.vocab_size = vocab_size
        self.ndim = ndim
        self.maxlen = maxlen

    def compile_model(self, filters=32, window=7, pool_size=5, lr=1e-4) -> models.Sequential:
        """CNN layer model definition."""
        model = models.Sequential(name="ConvNet")
        model.add(layers.Embedding(self.vocab_size, self.ndim, input_length=self.maxlen))
        model.add(layers.Conv1D(filters, window, activation="relu"))
        model.add(layers.MaxPooling1D(pool_size))
        model.add(layers.Conv1D(filters, window, activation="relu"))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dense(1))
        model.compile(optimizers.RMSprop(lr=lr),
                      loss="binary_crossentropy",
                      metrics=["acc"])
        return model
