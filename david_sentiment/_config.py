from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

from david.tokenizers import Tokenizer
from keras.models import load_model

from .utils import INI_TEMPLATE_MODEL_CONFIG, INIFileConfig


def create_project_structure(config, clear_first=False, exist_ok=False):
    """Create root-directory, sub-directories and attach the config,
        model, vocab file paths.

    `config`: Class instance of `YTSentimentConfig`.
    `clear_first`: Whether to clear the paths in place (if any)
        before attaching the created paths.
    """
    output_dir = Path(config.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=exist_ok)
    model_path = output_dir.joinpath("model")
    if not model_path.exists():
        model_path.mkdir(exist_ok=exist_ok)
    vocab_path = output_dir.joinpath("vocab")
    if not vocab_path.exists():
        vocab_path.mkdir(exist_ok=exist_ok)

    if clear_first:
        config.config_file = "config.ini"
        config.model_file = "model.h5"
        config.vocab_file = "vocab.pkl"

    config.output_dir = output_dir
    config.model_file = model_path.joinpath(config.model_file)
    config.vocab_file = vocab_path.joinpath(config.vocab_file)
    config.config_file = output_dir.joinpath(config.config_file)
    return config


@dataclass
class YTSentimentConfig:
    min_strlen: int = 20
    max_strlen: int = 400
    spacy_model: str = "en_core_web_sm"
    glove_ndim: str = "100d"
    epochs: int = 100
    trainable: bool = False
    padding: str = "post"
    loss: str = "binary_crossentropy"
    optimizer: str = "adam"
    activation: str = "sigmoid"
    output_dir: str = "sentiment_model"
    model_file: str = "model.h5"
    vocab_file: str = "vocab.pkl"
    config_file: str = "config.ini"
    max_seqlen: int = None
    vocab_shape: Tuple[int, int] = None

    def _init_project(self, clear_first=False, exist_ok=False, template=None):
        self = create_project_structure(self, clear_first, exist_ok=exist_ok)
        if template is None:
            template = INI_TEMPLATE_MODEL_CONFIG
        if not hasattr(template, "format_map"):
            raise AttributeError(
                f"template {type(template)} has no attribute to format_map."
            )
        config_template = template.format_map(asdict(self))
        INIFileConfig(self.config_file, config_template, exist_ok=exist_ok)

    def _load_model_and_tokenizer(self):
        """Restore/load the model and tokenizer objects from a saved file."""
        self.model = load_model(self.model_file)
        self.tokenizer = Tokenizer(self.vocab_file)

    @staticmethod
    def load_project(filename: str = None):
        """Load an existing project from a initialization file."""
        if filename is None:
            filename = YTSentimentConfig.config_file
        kwargs = {}
        for section in INIFileConfig(filename).values():
            for arg, value in section.items():
                kwargs[arg] = value
        return YTSentimentConfig(**kwargs)

    def save_project_files(self, clear_filepaths=False, exist_ok=False):
        """Save all configurations and create directories and files."""
        self._init_project(clear_first=clear_filepaths, exist_ok=exist_ok)
        self.model.save(self.model_file)
        self.tokenizer.save_vectors(self.vocab_file)
