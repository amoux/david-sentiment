from dataclasses import asdict
from os.path import isfile
from typing import List, Sequence

from david.models import GloVe
from david.text import largest_sequence
from david.tokenizers import Tokenizer
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from ._config import YTCSentimentConfig
from .utils import nearest_emoji


class YTCSentimentModel(YTCSentimentConfig):
    __config_loaded_from_file = False

    def __init__(self, model_config=None, **kwargs):
        if model_config is not None:
            if isinstance(model_config, YTCSentimentConfig):
                if isfile(model_config.config_file):
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
        self.sequences = None
        if self.__config_loaded_from_file:
            self._load_model_and_tokenizer()

    def _build_vocabulary(self, train_data):
        self.tokenizer.fit_on_document(train_data)
        self.tokenizer.vocabulary_to_frequency(self.mintoken_freq)
        sequences = self.tokenizer.document_to_sequences(train_data)
        self.max_seqlen = largest_sequence(sequences)
        self.sequences = sequences

    def _compile_model(self, train_data):
        self._build_vocabulary(train_data)
        model = Sequential()
        vocab_embeddings = GloVe.fit_embeddings(
            self.tokenizer.vocab_index, vocab_dim=self.glove_ndim
        )
        self.vocab_shape = vocab_embeddings.shape
        embedding_layer = Embedding(
            self.vocab_shape[0],
            self.vocab_shape[1],
            input_length=self.max_seqlen,
            weights=[vocab_embeddings],
            trainable=self.trainable,
        )
        model.add(embedding_layer)
        model.add(Flatten())
        model.add(Dense(1, activation=self.activation))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["acc"])
        self.model = model

    def train_model(self, train_data: List[str], train_labels: List[int]):
        self._compile_model(train_data)
        padded_sequences = pad_sequences(self.sequences, padding=self.padding)
        self.model.fit(padded_sequences, train_labels, epochs=self.epochs, verbose=1)

    def pad_input(self, text: str) -> List[Sequence[int]]:
        embedd = self.tokenizer.convert_string_to_ids(text)
        return pad_sequences([embedd], maxlen=self.max_seqlen, padding=self.padding)

    def predict(self, text: str, k=0.6) -> float:
        embedd_input = self.pad_input(text)
        embedd_score = self.model.predict(embedd_input)[0]
        rounded_score = round(embedd_score[0] * 100, 4)
        if embedd_score[0] >= k:
            return (1, rounded_score)
        else:
            return (0, rounded_score)

    def print_predict(self, text: str, k=0.6) -> None:
        label, score = self.predict(text, k=k)
        emoji = nearest_emoji(score)
        output = 'input: "{}" < {} ({})% >'
        if label == 1:
            output = output.format(text, f"pos({emoji})", score)
        else:
            output = output.format(text, f"neg({emoji})", score)
        print(output)

    def __repr__(self):
        return "<YTCSentimentModel(max_seqlen={}, vocab_shape={})>".format(
            self.max_seqlen, self.vocab_shape
        )
