import configparser
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from david.text import normalize_whitespace

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
[Corpus]
max_strlen: {max_strlen}
min_strlen: {min_strlen}
spacy_model: {spacy_model}
[Tokenizer]
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
output_dir: {output_dir}
model_file: {model_file}
vocab_file: {vocab_file}
config_file: {config_file}"""


def nearest_emoji(score: float) -> str:
    """Find the nearest emoji matching a sentiment value."""
    array = np.asarray(list(EMOJI_EMOTIONS.keys()))
    index = (np.abs(array - score)).argmin()
    emoji_index = array[index]
    if emoji_index:
        return EMOJI_EMOTIONS[emoji_index]
    return "⛔"


def test_polarity_attention_weights(model):
    face = {"pos": ":)", "neg": ":("}
    positive_negative = "I love this, but hate it {}"
    negative_positive = "I hate this, but love it {}"
    model.print_predict(negative_positive.format(face["pos"]))
    model.print_predict(positive_negative.format(face["pos"]))
    model.print_predict(negative_positive.format(face["neg"]))
    model.print_predict(positive_negative.format(face["neg"]))


def test_unseen_samples(
        model, test_data: List[Tuple[str, float]], print_k: int):
    for y_text, y_score in random.sample(test_data, k=print_k):
        _, x_score = model.predict(y_text)
        emoji = nearest_emoji(x_score)
        text = normalize_whitespace(y_text)
        print(f"💬 (Old={y_score}, New={x_score})\n {emoji} - {text}\n")


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
                raise FileExistsError(f"File {filename} exists. Set "
                                      "exist_ok as True, to override file.")
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
