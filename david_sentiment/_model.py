from dataclasses import asdict
from os import path
from typing import Any, Iterable, List, Sequence, Tuple, Dict

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
        """Load the model from a configuration instance."""
        if model_config is not None:
            if isinstance(model_config, SentimentConfig):
                if path.isfile(model_config.config_file):
                    self.__config_loaded_from_file = True
                kwargs = asdict(model_config)

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
        self.embedding_layer = None
        if self.__config_loaded_from_file:
            self._load_model_and_tokenizer()

    def embedding(self, mask_zero=False, l2=1e-6, trainable=False, ndim: str = None):
        """Fit the vocabulary to glove's embedding vectors.

        `mask_zero`: Note, masking is not supported for `Flatten()` layers.
        `trainable`: Whether to freeze the embedding layer. When parts of the model
            are pre-trained (embedding-layer), and parts are randomly initialized
            like (classifier), the pre-trained should not be updated at training
            to avoid forgetting what the weights already know. The large gradient
            update triggered by the randomly initialized layers would be very
            disruptive to the already leaned features.
        """
        if not trainable:
            trainable = self.trainable
        if ndim is None:
            ndim = self.glove_ndim

        vocab_index = self.tokenizer.vocab_index
        vocab_matrix = GloVe.fit_embeddings(vocab_index, ndim)
        l2_regulizer = L1L2(l2=l2) if l2 != 0 else None
        embedding_layer = Embedding(
            name="embedding",
            input_dim=vocab_matrix.shape[0],
            output_dim=vocab_matrix.shape[1],
            weights=[vocab_matrix],
            mask_zero=mask_zero,
            input_length=self.max_seqlen,
            trainable=trainable,
        )
        self.vocab_shape = vocab_matrix.shape
        return embedding_layer

    def _compile_network(
        self, task="pre-trained", model=None, layer=None, return_model=False
    ):
        """Compile a network to the model for a specific learning task.

        `task`: Learning modes for two different tasks:

            `pre-trained` - learning from pre-trained embeddings (default - mode)
            `ad-hoc`      - learning a task-specific embedding from the document's vocabulary.
        """

        def pretrained(model, layer):
            model.add(layer)
            model.add(Flatten())
            model.add(Dense(32, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(optimizer="rmsprop",
                          loss="binary_crossentropy",
                          metrics=["acc"])
            return model

        def adhoc(model, layer):
            maxtoks, ndim, maxlen = layer
            model.add(Embedding(maxtoks, ndim, input_length=maxlen))
            model.add(Flatten())
            model.add(Dense(32, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(optimizer="rmsprop",
                          loss="binary_crossentropy",
                          metrics=["acc"])
            return model

        if task == "pre-trained":
            if layer is not None and isinstance(layer, Embedding):
                model = pretrained(Sequential() if model is None else model, layer)
        elif task == "ad-hoc":
            if layer is not None and len(layer) == 3:
                model = adhoc(Sequential() if model is None else model, layer)
        if not return_model:
            self.model = model
        else:
            return model

    def compile_network(self, model=None, layer=None, return_model=False):
        """Compile the embedding model (returns model or takes place in-place)."""
        if model is None:
            model = Sequential()
        if layer is None:
            layer = self.embedding()

        model.add(layer)
        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
        if return_model:
            return model
        else:
            self.model = model

    def transform(
        self, x=None, y=None, segment=False, split_ratio=0.2, mincount: int = None
    ):
        """Transform texts and its labels to (x1, y1), (x2, y2)."""
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
            raise ValueError("Please check your objects.")

        if mincount is None:
            mincount = self.min_vocab_count

        split_ratio = int(split_ratio * int(len(labels)))
        x_train = data[:-split_ratio]
        y_train = labels[:-split_ratio]
        x_test = data[-split_ratio:]
        y_test = labels[-split_ratio:]

        self.tokenizer.fit_on_document(document=data)
        self.tokenizer.fit_vocabulary(mincount=mincount)
        y_train = np.asarray(y_train).astype("int32")
        y_test = np.asarray(y_test).astype("int32")
        x_train = self.tokenizer.document_to_sequences(x_train)
        x_test = self.tokenizer.document_to_sequences(x_test)

        self.max_seqlen = largest_sequence(x_train)
        x_train = pad_sequences(x_train, self.max_seqlen, padding=self.padding)
        x_test = pad_sequences(x_test, self.max_seqlen, padding=self.padding)
        return x_train, y_train, x_test, y_test

    def train(
        self, x=None, y=None, epochs=None, segment=True, split_ratio=0.2, batch_size=32,
    ):
        """Initialize the training session from texts and labels.

        NOTE: Model will be trained in place. Calls `self.transform()`,
            `self.compile_network()` and `self.model.fit()` sequentially.
        """
        if epochs is None:
            epochs = self.epochs

        x1, y1, x2, y2 = self.transform(x, y, segment, split_ratio)
        self.compile_network()
        self.model.fit(x1, y1,
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=1,
                       validation_data=(x2, y2),)

    def _validate(self, x_val: List[str], y_val: List[int]):
        # this should be in the train method. With a parameter to choice
        # whether to return a callable object with the x_test, y_test data
        # sets loaded. So the user can simply call the method on his own.
        # Since (x1, y1), (x2, y2) is done internally.
        if not isinstance(y_val, np.ndarray):
            y_val = np.asarray(y_val).astype("int32")

        x_val = self.tokenizer.document_to_sequences(document=x_val)
        x_val = pad_sequences(x_val, self.max_seqlen, padding=self.padding)
        loss, acc = self.model.evaluate(x_val, y_val)
        print(f"loss: {loss}\nacc: {acc}")

    def encode(self, sequence: str) -> List[Sequence[int]]:
        """Pad an input string sequence based on the model's padding."""
        embedd = self.tokenizer.convert_string_to_ids(sequence)
        max_seqlen = self.max_seqlen
        if not isinstance(max_seqlen, int):
            max_seqlen = int(max_seqlen)
        return pad_sequences([embedd], maxlen=max_seqlen, padding=self.padding)

    def predict(self, sequence: str, k=0.6) -> float:
        """Predict the sentiment value for a given string."""
        embedd_input = self.encode(sequence)
        embedd_score = self.model.predict(embedd_input)[0]
        rounded_score = round(embedd_score[0] * 100, 4)
        if embedd_score[0] >= k:
            return (1, rounded_score)
        else:
            return (0, rounded_score)

    def print_predict(self, text: str, k=0.6) -> None:
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
        """Print the instance PEP8."""
        return "<SentimentModel(max_seqlen={}, vocab_shape={})>".format(
            self.max_seqlen, self.vocab_shape
        )
