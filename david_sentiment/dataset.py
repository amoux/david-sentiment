from dataclasses import dataclass, field
from functools import partial
from types import GeneratorType
from typing import Any, Iterable, List, NewType, Set, Tuple, Union

import numpy as np
import spacy
from david.datasets import YTCommentsDataset as YTComments
from david.server import CommentsSql
from david.text import (normalize_whitespace, remove_punctuation,
                        unicode_to_ascii)
from textblob import TextBlob
from tqdm import tqdm
from wasabi import msg

_SentimentConfig = NewType("SentimentConfig", object)
_Trainable = NewType("TrainableDataset", List[Tuple[str, int]])
_Untrainable = NewType("UntrainableDataset", List[Tuple[str, float]])
_Tx = List[Tuple[Any, int]]
_Ts = Tuple[Iterable[Any], Iterable[int]]


@dataclass
class Fetch:
    """Single query from a string."""

    db: str
    q: str


@dataclass
class FetchMany:
    """Multiple queries from a set."""

    db: str
    q: Set[str]


@dataclass
class BatchDB:
    """Query a list of Fetch and FetchMany objects."""

    queries: List[Union[Fetch, FetchMany]]


def fetch_queries(db_batch: BatchDB) -> List[str]:
    """Build a batch of texts from single or/and many database queries."""
    if not isinstance(db_batch, BatchDB):
        raise TypeError("db_batch is expected to be an instance of DatabaseBatch")

    batches: List[str] = []
    for batch in db_batch.queries:
        if isinstance(batch, Fetch):
            db = CommentsSql(batch.db)
            comments = db.fetch_comments(batch.q)
            for comment in comments:
                if comment.text in batches:
                    continue
                batches.append(comment.text)

        if isinstance(batch, FetchMany):
            db = CommentsSql(batch.db)
            for query in batch.q:
                comments = db.fetch_comments(query)
                for comment in comments:
                    if comment.text in batches:
                        continue
                    batches.append(comment.text)

    return batches


def segment_binary_dataset(data: Union[_Tx, _Ts], unpack=False):
    """Segment a dataset's binary boundries to 1:1 distributions."""
    if isinstance(data, tuple):
        try:
            x, y = data
        except ValueError:
            raise Exception("Tuple must have two iterables, got 1")
        else:
            data = list(zip(x, y))

    isinteger = isinstance(data[0][1], int)
    assert isinteger == 0 or isinteger == 1, (
        "Integer items should be 1 or 0, got {}".format(data[0][1]))

    def segment(x: Any, z: int, o: int) -> Iterable[Any]:
        return x[round(z - o) :] if z > o else x[: round(z - o)]

    data = sorted(data, key=lambda k: k[1])
    data, base2 = zip(*data)
    zeros = base2.count(0)
    ones = base2.count(1)
    data = segment(data, zeros, ones)
    base2 = segment(base2, zeros, ones)

    assert base2.count(0) == base2.count(1)
    dataset = list(zip(data, base2))
    np.random.shuffle(dataset)
    if unpack:
        data, base2 = zip(*dataset)
        return data, base2
    return dataset


def preprocess_string(sequence: str) -> str:
    """Encode ascii, remove punctuation, normalize-whitespaces."""
    return normalize_whitespace(unicode_to_ascii(sequence))


def _meta_learner(sequence: str, preprocessor=None, learner=None) -> float:
    if not isinstance(sequence, str):
        raise TypeError(f"Invalid sequence: {type(sequence)}, must be of str type.")
    if preprocessor is not None:
        sequence = preprocessor(sequence)
    if learner is not None:
        sequence = remove_punctuation(sequence)
        sentiment = learner(sequence)
        if hasattr(sentiment, "sentiment"):
            sentiment = sentiment.sentiment

    return sentiment


def textblob_meta_learner(document: List[str], return_untrainable=False):
    """Textblob sentiment scores for automatic data annotation.

    - Binary rule: `k = 0 if polarity < .0 else 1`
        - `∀x(0, ...) ∩ (..., 1)`
        - `[-1.0, -0.9, ..., -1.0] ∩ [0.1, 0.5, ..., 1.0]`

    Learner:
    --------
        - negative|positive : -1.0 | 1.0 (trainable)
        - neutral  : 0.0 (untrainable)

    Usually, the output scores from the sentiment method are
    intervals between `[.0, .., .1]`. This method only cares
    whether a predicted value is close to either 0 or 1
    (negative or positive). Thus, only included as intervals
    `[0, 1]` for each string sequence and if output is `0.0`
     then, it gets added separately.
    """
    msg.warn(f" TextBlob annotating dataset with binary classes: 0 | 1")

    annotator = partial(_meta_learner, preprocessor=preprocess_string, learner=TextBlob)
    untrainable, trainable = [], []
    for text in document:
        sentiment = annotator(text)
        polarity = sentiment.polarity
        if polarity == 0.0:
            untrainable.append((text, polarity))
        else:
            k = 0 if polarity < 0.0 else 1
            trainable.append((text, k))

    msg.good(f" annotation summary 🤖")
    msg.info(f" trainable: {len(trainable)}, un-trainable: {len(untrainable)}")
    if return_untrainable:
        return trainable, untrainable
    return trainable


def batch_to_texts(batch: List[str], maxlen: int) -> List[str]:
    """Transform a batch to an iterable of string sequences (texts).

    NOTE: Filtering of small strings is done in the `texts_to_sents()`
    method. Since the strings are tokenized to sentences. Lastly, any
    existing (duplicated) strings are skipped.

    `maxlen`: Skip any strings of length greater than a given maximum value.
    """
    batch_size = len(batch)
    msg.warn(f" normalizing {batch_size} string sequences from batch.")

    texts: List[str] = []
    for string in tqdm(batch, desc="sequences", unit=""):
        string = preprocess_string(string)
        if len(string) <= maxlen and string not in texts:
            texts.append(string)

    large_strings = batch_size - len(texts)
    msg.good(f" removed {large_strings} strings of length >= {maxlen}.")
    msg.info(f" returning batch with {len(texts)} samples.")
    return texts


def texts_to_sents(texts: List[str], minlen: int, spacy_model="en_core_web_sm"):
    """Transform texts to to sentences using spaCy's sentence tokenizer.
    
    `mincount`: Skip any strings of length less than a given minimum value. 
    """
    nlp = spacy.load(spacy_model)
    msg.warn(f" tokenizing strings to sentences, spaCy model: {spacy_model}.")

    sentences: List[str] = []
    small_strings = 0
    for doc in tqdm(nlp.pipe(texts), desc="sequences", unit=""):
        for sent in doc.sents:
            text = sent.text
            if len(text) <= minlen:
                small_strings += 1
                continue
            sentences.append(text)

    msg.good(f" removed {small_strings} strings of length <= {minlen}.")
    msg.info(f" previous-size: {len(texts)}, new-size: {len(sentences)}.")
    return sentences


def build_dataset(
    batch: Union[BatchDB, GeneratorType, List[str]],
    config: _SentimentConfig,
    untrainable=False,
) -> Union[_Trainable, _Untrainable]:
    """Build a training dataset from db batch or an iterable of strings.

    `untrainable`: Whether to return the dataset with no polarity labels
        along with the trainable dataset. If true, two iterable of tuples
        will be returned -> [(texts, labels)], [(texts, labels)]
    """
    min_strlen = int(config.min_strlen)
    max_strlen = int(config.max_strlen)
    spacy_model = str(config.spacy_model)

    if isinstance(batch, BatchDB):
        batch = fetch_queries(batch)
    elif isinstance(batch, GeneratorType):
        batch = list(batch)
    elif not isinstance(batch, list):
        raise ValueError(
            f"Invalid batch {type(batch)}. Must be an instance"
            " of BatchDB, GeneratorType or List[str]"
        )
    texts = batch_to_texts(batch, maxlen=max_strlen)
    sents = texts_to_sents(texts, minlen=min_strlen, spacy_model=spacy_model)
    return textblob_meta_learner(sents, untrainable)
