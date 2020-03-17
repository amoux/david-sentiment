from dataclasses import asdict
from os import path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from david.models import GloVe
from david.text import largest_sequence
from david.tokenizers import Tokenizer
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import L1L2

from ._config import SentimentConfig
from .dataset import segment_binary_dataset
from .utils import nearest_emoji


class SentimentModel(SentimentConfig):
    """Sentiment model class."""

    __config_loaded_from_file = False

    def __init__(self, model_config=None, **kwargs):
        """Load the model from a configuration.

        `model_config`: A `path/to/config.ini` or an instance of `SentimentConfig`.
        """
        if model_config is not None:
            if isinstance(model_config, SentimentConfig):
                if path.isfile(model_config.config_file):
                    self.__config_loaded_from_file = True
                kwargs = asdict(model_config)
            elif isinstance(model_config, str):
                if path.isfile(model_config):
                    self.__config_loaded_from_file = True
                kwargs = asdict(self.load_project(model_config))

        super().__init__(**kwargs)
        self.tokenizer = Tokenizer(
            remove_urls=self.remove_urls,
            enforce_ascii=self.enforce_ascii,
            reduce_length=self.reduce_length,
            preserve_case=self.preserve_case,
            strip_handles=self.strip_handles,
        )
        self.model = None
        self.vocab_matrix = None
        if self.__config_loaded_from_file:
            self._load_model_and_tokenizer()
            # the tokenizer needs to be loaded
            # by calling the following method.
            self.tokenizer.fit_vocabulary()

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Return the input shape representation: `(vocab_size, ndim, max_seqlen)`."""
        ndim = int(self.ndim.replace('d', ''))
        return (self.vocab_size, ndim, self.max_seqlen)

    def embedding(self, module: str = None, ndim: str = None, l2=1e-6):
        """Build the embedding layer with GloVe vectors.

        - The embedding layer can be built with one of the following:

            - module: `6b`    -> glove.6B | `'50d', '200d', '100d', '300d'`
            - module: `27b`   -> glove.twitter.27B | `'25d', '50d', '100d', '200d'`
            - module: `42b`   -> glove.42B | `'300d'`
            - module: `840b`  -> glove.840B | `'300d'`

        """
        if ndim is not None:
            self.ndim = ndim
        else:
            ndim = self.ndim

        glove_module = None
        if module is not None and isinstance(module, str):
            if module.lower() in GloVe.modules:
                glove_module = GloVe(module)
            else:
                modules = set(GloVe.modules.keys())
                raise ValueError(f"'{module}' not in {modules}.")
        else:
            raise TypeError("Module name needs to be of str type "
                            "found, {}.".format(type(module)))
    
        name = f"GloVe-{module.upper()}-{ndim}"
        matrix = glove_module(ndim=ndim, vocab=self.tokenizer.vocab_index)
        l2_regulizer = L1L2(l2=l2) if l2 != 0 else None
        embedding_layer = Embedding(name=name,
                                    input_dim=matrix.shape[0],
                                    output_dim=matrix.shape[1],
                                    weights=[matrix],
                                    embeddings_regularizer=l2_regulizer,
                                    mask_zero=False,
                                    input_length=self.max_seqlen,
                                    trainable=False)

        self.vocab_size = matrix.shape[0]
        return embedding_layer

    def compile_net(self, model=None, layer=None, mode="pre-trained", return_model=True):
        """Compile a network to the model for a specific learning task.

        `mode`: learning mode for a type of task.

        - learning modes:
            - `pre-trained` : learning from pre-trained vectors (default).
            - `ad-hoc`      : learning from document's vectors, task-specific.
        """

        def pretrained(model, layer):
            model.name = "david-sentiment (PT)"
            model.add(layer)
            model.add(Flatten())
            model.add(Dense(32, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(optimizer="rmsprop",
                          loss="binary_crossentropy",
                          metrics=["acc"])
            return model

        def adhoc(model, layer):
            model.name = "david-sentiment (AH)"
            vocab_size, ndim, max_seqlen = layer
            model.add(Embedding(vocab_size, ndim, input_length=max_seqlen))
            model.add(Flatten())
            model.add(Dense(32, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(optimizer="rmsprop",
                          loss="binary_crossentropy",
                          metrics=["acc"])
            return model

        if mode.lower() == "pre-trained":
            if layer is not None and isinstance(layer, Embedding):
                model = pretrained(Sequential() if model is None else model, layer)

        elif mode.lower() == "ad-hoc":
            if layer is not None and len(layer) == 3:
                model = adhoc(Sequential() if model is None else model, layer)

        if not return_model:
            self.model = model
        else:
            return model

    def transform(self, x=None, y=None, mincount: int = None,
                  split_ratio=0.2, segment=True, test_vocab=None):
        """Transform texts and its labels to (x1, y1), (x2, y2).

        `test_vocab`: Iterable document of strings to extend the vocabulary.
        """
        if x is not None and y is None:
            if segment:
                data, labels = segment_binary_dataset(x, True)
            elif isinstance(x, list):
                data, labels = zip(*x)
            elif isinstance(x, tuple):
                data, labels = x
        elif (x and y) is not None:
            if segment:
                data, labels = segment_binary_dataset((x, y), True)
            data, labels = x, y
        else:
            raise ValueError("Data is not an iterable of list or tuple"
                             f", got {type(x or y)}.")
        if mincount is None:
            mincount = self.mincount

        split_ratio = int(split_ratio * len(labels))
        x_train = data[:-split_ratio]
        y_train = labels[:-split_ratio]
        x_test = data[-split_ratio:]
        y_test = labels[-split_ratio:]

        if test_vocab is not None:
            if isinstance(test_vocab[0], tuple):
                data2, _ = zip(*test_vocab)
            data = data + data2

        self.tokenizer.fit_on_document(document=data)
        self.tokenizer.fit_vocabulary(mincount=mincount)
        y_train = np.asarray(y_train).astype("int32")
        y_test = np.asarray(y_test).astype("int32")
        x_train = self.tokenizer.document_to_sequences(x_train)
        x_test = self.tokenizer.document_to_sequences(x_test)

        self.max_seqlen = largest_sequence(x_train)
        x_train = pad_sequences(x_train, self.max_seqlen, padding=self.padding)
        x_test = pad_sequences(x_test, self.max_seqlen, padding=self.padding)

        self.vocab_size = self.tokenizer.vocab_size + 1
        return x_train, y_train, x_test, y_test

    def train(self,
              x=None,
              y=None,
              epochs: int = None,
              batch_size=32,
              mincount: int = None,
              split_ratio=0.2,
              segment=True,
              mode="pre-trained",
              module="6b",
              ndim: str = None,
              l2=1e-6,
              return_datasets=False):
        """Initialize training - transforming texts, labels to arrays and compiling the model.

        Arguments:
        ----------
            x: Either a `List[str]` of strings or `List[Tuple[List[str], List[int]]]`
                where `List[str]` are strings and `List[int]` are binary labels of `0|1`.
            y: If `x` argument is a list of strings then `y` is a list of binary integers.
            epochs: Number of iterarations per batch.
            mincount: Reduce the number of tokens with a min-count/frequecy of N.
            mode: Whether to train the model as `'pre-trained' or 'ad-hoc'`.
            return_dataset: Whether to return the train and test sets `x1, y1, x2, y2`.
        """
        if epochs is not None and isinstance(epochs, int):
            self.epochs = epochs
        elif ndim is not None and isinstance(ndim, str):
            self.ndim = ndim

        x1, y1, x2, y2 = self.transform(x, y, mincount, split_ratio, segment)

        if mode == "pre-trained":
            layer = self.embedding(module, ndim=self.ndim, l2=l2)
        elif mode == "ad-hoc":
            layer = self.input_shape

        self.compile_net(None, layer=layer, mode=mode, return_model=False)
        self.model.fit(x1, y1,
                      epochs=self.epochs,
                      batch_size=batch_size,
                      verbose=1,
                      validation_data=(x2, y2))

        if return_datasets:
            return x1, y1, x2, y2

    def evaluate(self, x_test, y_test, model=None, transform=False):
        """Evaluate the test sets (texts, labels).

        This method is the same as using `model.evaluate(x2, y2)`.

        `transform`: tranform integer labels to numpy arrays, convert string
            sequences to sequences of integers (int32) and pad the sequences.
            (note: this assumes the texts and labels have not been transformed
            already. otherwise leave as false).
        """
        if transform:
            if not isinstance(y_test, np.ndarray):
                y_test = np.asarray(y_test).astype("int32")
            x_test = self.tokenizer.document_to_sequences(x_test)
            x_test = pad_sequences(x_test, self.max_seqlen, "int32", self.padding)

        model = self.model if model is None else model
        loss, acc = model.evaluate(x_test, y_test)
        print(f"loss: {loss}\nacc: {acc}")

    def encode(self, sequence: str) -> List[Sequence[int]]:
        """Pad an input string sequence based on the model's padding."""
        embedd = self.tokenizer.convert_string_to_ids(sequence)
        max_seqlen = self.max_seqlen

        if not isinstance(max_seqlen, int):
            max_seqlen = int(max_seqlen)
        return pad_sequences([embedd], maxlen=max_seqlen, padding=self.padding)

    def predict(self, sequence: str, k=0.5, model=None) -> float:
        """Predict the sentiment value for a given string."""
        embedd_input = self.encode(sequence)
        model = self.model if model is None else model
        embedd_score = model.predict(embedd_input)[0]
        rounded_score = round(embedd_score[0] * 100, 4)

        if embedd_score[0] >= k:
            return (1, rounded_score)
        else:
            return (0, rounded_score)

    def predict_print(self, text: str, k=0.5) -> None:
        """Pretty print the prediction from a given string sequence."""
        label, score = self.predict(text, k=k)
        emoji = nearest_emoji(score)
        output = 'input: "{}" < {} ({})% >'
        if label == 1:
            output = output.format(text, f"pos({emoji})", score)
        else:
            output = output.format(text, f"neg({emoji})", score)
        print(output)

    def reset(self):
        """Explicitly reset the model and tokenizer."""
        self.tokenizer = Tokenizer(
            remove_urls=self.remove_urls,
            enforce_ascii=self.enforce_ascii,
            reduce_length=self.reduce_length,
            preserve_case=self.preserve_case,
            strip_handles=self.strip_handles,
        )
        self.model = Sequential()
        self.vocab_matrix = None

    @staticmethod
    def clone(sentiment: "SentimentModel", model=None) -> "SentimentModel":
        """Clone a SentimentModel class instance without the model attached."""
        sentiment_obj = SentimentModel()
        if isinstance(sentiment, SentimentModel):
            for key, value in vars(sentiment).items():
                if key == "model":
                    x = None if model is None else model
                    setattr(sentiment_obj, key, x)
                else:
                    setattr(sentiment_obj, key, value)

        return sentiment_obj

    def __repr__(self):
        return "<SentimentModel(vocab_size={}, ndim={}, max_seqlen={})>".format(
            self.vocab_size, self.ndim.replace("d", ""), self.max_seqlen)
