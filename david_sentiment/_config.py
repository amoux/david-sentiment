from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

from david.tokenizers import Tokenizer
from keras.models import load_model

from .utils import INI_TEMPLATE_MODEL_CONFIG, INIFileConfig

int_config_args = [
    "vocab_size",
    "max_seqlen",
    "mincount",
    "max_strlen",
    "min_strlen",
    "epochs",
]
bool_config_args = [
    "remove_urls",
    "enforce_ascii",
    "reduce_length",
    "preserve_case",
    "strip_handles",
]


def create_project_structure(config, clear_first=False, exist_ok=False):
    """Create root-directory, sub-directories and attach the config, model, vocab file paths.

    `config`: Class instance of `SentimentConfig`.
    `clear_first`: Whether to clear the paths in place (if any)
        before attaching the created paths.
    """
    project = Path(config.project)
    if not project.exists():
        project.mkdir(exist_ok=exist_ok)
    model_path = project.joinpath("model")
    if not model_path.exists():
        model_path.mkdir(exist_ok=exist_ok)
    vocab_path = project.joinpath("vocab")
    if not vocab_path.exists():
        vocab_path.mkdir(exist_ok=exist_ok)

    if clear_first:
        config.config_file = "config.ini"
        config.model_file = "model.h5"
        config.vocab_file = "vocab.txt"
        config.vectors_file = "vectors.pkl"

    config.project = project
    config.model_file = model_path.joinpath(config.model_file)
    config.vocab_file = vocab_path.joinpath(config.vocab_file)
    config.vectors_file = vocab_path.joinpath(config.vectors_file)
    config.config_file = project.joinpath(config.config_file)
    return config


@dataclass
class SentimentConfig:
    """Main configurator for preprocessing, tokenizer and embedding model.

    optimizers: `adam`, `rmsprop`
    """

    project: str = "sm-model"
    min_strlen: int = 20
    max_strlen: int = 1000
    mincount: int = 1
    epochs: int = 10
    ndim: str = "100d"
    spacy_model: str = "en_core_web_sm"
    remove_urls: bool = True
    enforce_ascii: bool = True
    reduce_length: bool = True
    preserve_case: bool = False
    strip_handles: bool = False
    padding: str = "post"
    optimizer: str = "adam"
    activation: str = "sigmoid"
    model_file: str = "model.h5"
    vocab_file: str = "vocab.txt"
    vectors_file: str = "vectors.pkl"
    config_file: str = "config.ini"
    max_seqlen: int = None
    vocab_size: int = None

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
        self.tokenizer = Tokenizer(self.vectors_file)

    @staticmethod
    def load_project(filename: str = None):
        """Load an existing project's state from a `config.ini` file."""
        if filename is None:
            filename = SentimentConfig.config_file

        kwargs = {}
        for section in INIFileConfig(filename).values():
            for arg, value in section.items():
                if arg in int_config_args:
                    kwargs[arg] = int(value)
                elif arg in bool_config_args:
                    kwargs[arg] = bool(value)
                else:
                    kwargs[arg] = value

        return SentimentConfig(**kwargs)

    def save_project(self, exist_ok=False):
        """Create project's file structure and save important session.

        dependencies for both  the Tokenizer's vocabulary and Keras's Model.
        """
        self._init_project(clear_first=exist_ok, exist_ok=exist_ok)
        self.model.save(self.model_file)
        self.tokenizer.save_vocabulary(self.vocab_file)
        self.tokenizer.save_vectors(self.vectors_file)
