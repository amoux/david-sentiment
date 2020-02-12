# david-sentiment

## Unsupervised sentiment models from YouTube Comments.

- Train a custom sentiment model with just a few lines of code - Making it easy to try different configurations or preprocessing techniques.

### Usage

- Create a new project.

```python
from david_sentiment import YTCSentimentConfig
config = YTCSentimentConfig(project_dir="my_model_dir", max_strlen=500)
```

- Build a large dataset from batch. Fetch based on keyword pattterns or complete sentences.

```python
import david_sentiment.dataset as ds
batch = ds.BatchDB([ds.Fetch('unbox', '%make a video%'),
                    ds.FetchMany('v1', {"%want to buy ____%", "%I want  ____%"}),])

x_train, x_labels, y_test = ds.fit_batch_to_dataset(batch, config=config)
```

- Train the embedding model.

```python
from david_sentiment import YTCSentimentModel
ytc_sentiment = YTCSentimentModel(config)
ytc_sentiment.train_model()
```

- Save the project: Call `save_project()` to create the project directories, saves all essential settings for initiating a previous state, including; the trained-model and tokenizer's vocab files:

  - config file         : `<project_name>/config.init`
  - trained model       : `<project_name>/model/model.h5`
  - tokenizer vocab     : `<project_name>/vocab/vocab.pkl`

```python
ytc_sentiment.save_project()
```
