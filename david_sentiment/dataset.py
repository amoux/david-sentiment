from dataclasses import dataclass
from typing import List, NewType, Set, Tuple, Union

import spacy
from david.server import CommentsSql
from david.text import (
    get_sentiment_polarity,
    normalize_whitespace,
    remove_punctuation,
    unicode_to_ascii,
)
from tqdm import tqdm
from wasabi import msg

_YTSentimentConfig = NewType("YTCSentimentConfig", object)


@dataclass
class Fetch:
    db: str
    q: str


@dataclass
class FetchMany:
    db: str
    q: Set[str]


@dataclass
class BatchDB:
    queries: List[Union[Fetch, FetchMany]]


def preprocess_texts(batch: List[str], minlen: int, maxlen: int) -> List[str]:
    """Normalize raw texts by reducing extra whitespaces and encoding ASCII."""
    batch_size = len(batch)
    msg.warn(f"* Preprocessing batch with {batch_size} samples...")
    texts = []
    for string in tqdm(batch, desc="Comments", unit=""):
        string = normalize_whitespace(unicode_to_ascii(string))
        if len(string) > minlen and len(string) < maxlen and string not in texts:
            texts.append(string)

    msg.good(
        f"* Removed {batch_size - len(texts)} comments"
        f"  from {batch_size}. Returning size: {len(texts)}"
    )
    return texts


def texts_to_sentences(texts: List[str], spacy_model="en_core_web_sm") -> List[str]:
    """Transform texts to to sentences using spaCy's sentence tokenizer."""
    nlp = spacy.load(spacy_model)
    msg.warn(f"* Transforming texts to sentences with {spacy_model} model.")
    sentences = []
    lines_count = 0
    for doc in tqdm(nlp.pipe(texts), desc="Docs", unit=""):
        for sent in doc.sents:
            text = sent.text
            polarity = get_sentiment_polarity(remove_punctuation(text))
            sentences.append((text, polarity))
            lines_count += 1
    msg.good(f"* Done! Sucessfully preprocessed {lines_count} sentences.")
    msg.info(f"* Size before: {len(texts)}, and after: {len(sentences)}.")
    return sentences


def load_training_data(
    sentences: List[str],
) -> Tuple[List[str], List[str], List[Tuple[str, float]]]:
    """Build the training dataset and sentiment labels.

    About:
        Textblob sentiment polarity metrics are used as
        Metalearning weights.
    """
    msg.warn(f"* Building to training data...")

    x_train, x_labels, y_tests = [], [], []
    for sentence, sentiment in sentences:
        if sentiment == 0.0:
            y_tests.append((sentence, sentiment))
        else:
            x_labels.append(1 if sentiment > 0 else 0)
            x_train.append(sentence)

    msg.good(f"* Done! x_train: {len(x_train)}, x_labels: {len(x_labels)}.")
    return (x_train, x_labels, y_tests)


def build_database_batch(db_batch: BatchDB) -> List[str]:
    """Build a batch of texts from single or/and many database queries."""
    if not isinstance(db_batch, BatchDB):
        raise ValueError("db_batch is expected to be an instance of DatabaseBatch")

    batches = []
    for batch in db_batch.queries:
        if isinstance(batch, Fetch):
            db = CommentsSql(batch.db)
            texts = [
                i.text for i in db.fetch_comments(batch.q)
                if i.text not in batches]
            batches.extend(texts)

        if isinstance(batch, FetchMany):
            db = CommentsSql(batch.db)
            for query in batch.q:
                texts = [
                    i.text for i in db.fetch_comments(query)
                    if i.text not in batches]
                batches.extend(texts)
    return batches


def fit_batch_to_dataset(
    batch: Union[BatchDB, List[str]], config: _YTSentimentConfig,
) -> Tuple[List[str], List[int], Tuple[str, float]]:
    """Fit a batch of query patterns and apply it to the pipeline.

    Returns the train samples, labels and test samples.
    """
    min_strlen = int(config.min_strlen)
    max_strlen = int(config.max_strlen)
    spacy_model = str(config.spacy_model)

    if isinstance(batch, BatchDB):
        batch = build_database_batch(batch)
    elif not isinstance(batch, list):
        raise ValueError(
            f"Invalid batch {type(batch)}. Must be an instance"
            "of BatchDB or an iterable strings of type: List[str]"
        )

    texts = preprocess_texts(batch, minlen=min_strlen, maxlen=max_strlen)
    sents = texts_to_sentences(texts, spacy_model=spacy_model)
    x_train, x_labels, y_test = load_training_data(sents)
    return x_train, x_labels, y_test

