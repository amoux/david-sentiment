import configparser
import random
from pathlib import Path
from typing import List, NewType, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from david.text import normalize_whitespace, remove_punctuation

_SentimentModel = NewType("SentimentModel", object)

# Custom dict of keys as distance measures,
# and values are the emojis representing K
# closest sentiment. Feel free to add more
# or/and set keys as floats e.g., 0.1 to 0.99
EMOJI_EMOTIONS = {
    99: "😍",
    95: "🤗",
    90: "😀",
    80: "😁",
    70: "😊",
    75: "😅",
    55: "😑",
    50: "😶",
    45: "😒",
    35: "😬",
    30: "😳",
    25: "😤",
    20: "😠",
    10: "😡",
    5: "🤬",
}

INI_TEMPLATE_MODEL_CONFIG = """
[Batch]
max_strlen: {max_strlen}
min_strlen: {min_strlen}
spacy_model: {spacy_model}
[Tokenizer]
remove_urls: {remove_urls}
enforce_ascii: {enforce_ascii}
reduce_length: {reduce_length}
preserve_case: {preserve_case}
strip_handles: {strip_handles}
min_vocab_count: {min_vocab_count}
max_seqlen: {max_seqlen}
glove_ndim: {glove_ndim}
vocab_shape: {vocab_shape}
[Model]
activation: {activation}
trainable: {trainable}
epochs: {epochs}
loss: {loss}
optimizer: {optimizer}
padding: {padding}
[Output]
project_dir: {project_dir}
model_file: {model_file}
vocab_file: {vocab_file}
vectors_file: {vectors_file}
config_file: {config_file}"""


def nearest_emoji(score: float) -> str:
    """Find the nearest emoji matching a sentiment value."""
    array = np.asarray(list(EMOJI_EMOTIONS.keys()))
    index = (np.abs(array - score)).argmin()
    emoji_index = array[index]
    if emoji_index:
        return EMOJI_EMOTIONS[emoji_index]
    return "⛔"


def plot_losses(history, fmt1="c>", fmt2="m", show=True, save=False, name="loss.png", dpi=300):
    """Plot the training and validation loss.

    `fmt1, fmt2`: A format string consisting of `fmt='[marker][line][color]'`
    """
    if hasattr(history, "history"):
        history = history.history

    acc = history["acc"]
    loss = history["loss"]
    val_acc = history["val_acc"]
    val_loss = history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, fmt1, label="training-loss")
    plt.plot(epochs, val_loss, fmt2, label="validation-loss")
    plt.title("training and validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    if save:
        plt.savefig(name, dpi=dpi)
    if show:
        plt.show()


def plot_accuracy(history, fmt1="c>", fmt2="m", show=True, save=False, name="acc.png", dpi=300):
    """Plot the training and validation loss.

    `fmt1, fmt2`: A format string consisting of `fmt='[marker][line][color]'`
    """
    if hasattr(history, "history"):
        history = history.history

    acc = history["acc"]
    val_acc = history["val_acc"]
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, fmt1, label="training-acc")
    plt.plot(epochs, val_acc, fmt2, label="validation-acc")
    plt.title("training and validation accuracy")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    if save:
        plt.savefig(name, dpi=dpi)
    if show:
        plt.show()


def INIFileConfig(filename: str, template: str = None, exist_ok=False):
    """Create initialization files from templates or load config from file.

    `filename`: The name of the ini configuration file.
    `template`: The template is assumed to be a mapped str object.
    `exist_ok`: An error is raised if `exist_ok=False` and file exist.
    """
    filename = Path(filename)
    config = configparser.ConfigParser()
    config.optionxform = str

    def write(filename):
        with filename.open("w", encoding="utf8") as f:
            config.read_string(template)
            config.write(f)

    if template is not None:
        if not exist_ok:
            if not filename.exists():
                write(filename)
            else:
                raise FileExistsError(
                    f"File {filename} exists. Set "
                    "exist_ok as True, to override file."
                )
        else:
            write(filename)
    else:
        template_lines = {}
        config.read(filename)
        for section in config.sections():
            template_lines[section] = {}
            if config.items(section) is None:
                continue
            for arg, value in config.items(section):
                template_lines[section][arg] = value
        return template_lines


def interactive_session(sentiment: _SentimentModel, stopflag="quit") -> None:
    """Test a trained model interactively with your inputs.

    `sentiment`: class instance of a SentimentModel `predict()` is called.
    """
    binary = {0: "<negative>", 1: "<positive>"}
    while True:
        chat = "\n* sentiment stats => emoji: {}, label: {}, confidence: {}%\n"
        input_text = input("input : ")
        if input_text.lower() != stopflag:
            label, score = sentiment.predict(input_text)
            print(chat.format(nearest_emoji(score), binary[label], round(score, 2)))
        elif input_text.lower() == stopflag:
            break
        else:
            continue


def test_unseen_samples(
    sentiment: _SentimentModel,
    test_data: Union[List[Tuple[str, float]], List[str]],
    k: int = 10,
) -> None:
    """Predict sentiment scores on a test dataset with texts with/or without scores."""
    if isinstance(test_data, list):
        if not isinstance(test_data[0], tuple):
            test_data = list(zip(test_data, len(test_data) * [0.0]))

    for text, y_score in random.sample(test_data, k=k):
        label, x_score = sentiment.predict(text)
        emoji = nearest_emoji(x_score)
        text = normalize_whitespace(text)
        print(f"💬 <old={y_score}, new={x_score}>\n {emoji} - {text}\n")


def test_polarity_distance(sentiment: _SentimentModel, ntest=1) -> None:
    """Perform a simple test on words:`love` and `hate` at different positions.

    `ntest`: Up-to five different tests `ntest=5` prints all tests.
    """
    face = {"pos": ":)", "neg": ":("}
    emoji = {"pos": "😍", "neg": "😡"}
    textA = "I love this, but hate it"
    textB = "I hate this, but love it"
    textC = normalize_whitespace(remove_punctuation(textA))
    textD = normalize_whitespace(remove_punctuation(textB))
    # alway print some result for the friendly user.
    ntest = (5 or 1) if (ntest < 1 or ntest > 5) else ntest
    if ntest >= 1:
        print("* -- sentiment with punctuation -- *\n")
        sentiment.print_predict("{} {}".format(textB, face["pos"]))
        sentiment.print_predict("{} {}".format(textA, face["pos"]))
        sentiment.print_predict("{} {}".format(textB, face["neg"]))
        sentiment.print_predict("{} {}".format(textA, face["neg"]))
    if ntest >= 2:
        print("\n* -- sentiment without punctuation -- *\n")
        sentiment.print_predict("{} {}".format(textD, face["pos"]))
        sentiment.print_predict("{} {}".format(textC, face["pos"]))
        sentiment.print_predict("{} {}".format(textD, face["neg"]))
        sentiment.print_predict("{} {}".format(textC, face["neg"]))
    if ntest >= 3:
        print("\n* -- sentiment without emoticons -- *\n")
        sentiment.print_predict("{}".format(textD))
        sentiment.print_predict("{}".format(textC))
        sentiment.print_predict("{}".format(textD))
        sentiment.print_predict("{}".format(textC))
    if ntest >= 4:
        print("\n* -- sentiment with emoji's -- *\n")
        sentiment.print_predict("{} {}".format(textB, emoji["pos"]))
        sentiment.print_predict("{} {}".format(textA, emoji["pos"]))
        sentiment.print_predict("{} {}".format(textB, emoji["neg"]))
        sentiment.print_predict("{} {}".format(textA, emoji["neg"]))
    if ntest >= 5:
        print("\n* -- sentiment with emoji's & no punctuation -- *\n")
        sentiment.print_predict("{} {}".format(textD, emoji["pos"]))
        sentiment.print_predict("{} {}".format(textC, emoji["pos"]))
        sentiment.print_predict("{} {}".format(textD, emoji["neg"]))
        sentiment.print_predict("{} {}".format(textC, emoji["neg"]))
