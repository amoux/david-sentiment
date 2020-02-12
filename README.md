# david-sentiment

Unsurpervised sentiment models from **YouTube** Comments. Keep track of different configurations by passing a new name to the `project_dir` parameter.

## Usage

- Train a new model from scratch.

```python
from david_sentiment import YTCSentimentConfig

config = YTCSentimentConfig(project_dir="my_model_dir", max_strlen=500)
```

- Build a large dataset  from batch. Fetch based on keyword pattterns or complete sentences.

```python
import david_sentiment.dataset as ds

batch = ds.BatchDB([
    ds.Fetch('unbox', '%make a video%'),
    ds.FetchMany('v1', {"%want to buy ____%", "%I want  ____%"}),])

x_train, x_labels, y_test = ds.fit_batch_to_dataset(batch, config=config)
```

- Train the model on the `n` epochs.

```python
from david_sentiment import YTCSentimentModel

ytc_sentiment = YTCSentimentModel(config)
ytc_sentiment.train_model()
```

- Save the `Embedding` model and `Tokenizer` vocabulary vectors to file. These files can the be used to restore the ***"session"*** and full functionality. By calling `self.save_project()`:

  - Config-file         : `<project_name>/config.init`
  - Trained-model       : `<project_name>/model/model.h5`
  - Tokenizer-vocab     : `<project_name>/vocab/vocab.pkl`

```python
# project structure, config.ini
ytc_sentiment.save_project()
```
