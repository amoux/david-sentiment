# david-sentiment

## Unsupervised Sentiment Models

> Why *sentiment models from youtube comments?* - Because Twitter text-datasets are overrated, and lack ***sentimentalism***; *The excessive expression of feelings of tenderness, sadness, or nostalgia in behavior, writing, or speech.*

## Todo

- Needed features:
  - Multi-head attention (Transformer model)
  - Encoder/Decoder masking
  - Text generation (RNN)

- Train a custom sentiment model with just a few lines of code - Making it easy to try different configurations or preprocessing techniques.

### Usage

- Create a new project

New tokenizer preprocessing features:

- `enforce_ascii` : Keep only printable chars are added including emojis.
- `remove_urls`   : Remove all urls from the vocab.

```python
from david_sentiment import SentimentConfig
config = SentimentConfig(project_dir="ytc_sentiment",
                            max_strlen=3000,
                            epochs=100,
                            enforce_ascii=True, # :new
                            remove_urls=True,   # :new
                            glove_ndim="100d",)  
```

Build a dataset from database query patterns. Fetch based on keyword patterns or complete sentences.

```python
import david_sentiment.dataset as ds

batch = ds.BatchDB([ds.Fetch('unbox', "%make a video%"),
                    ds.Fetch('v1', "%make a video%"),])

x_train, x_labels, y_test = ds.build_dataset(batch, config)
```

> **NOTE**: Now it also work's for with any iterable document of strings `List[str]`

```python
from david.datasets import YTCommentsDataset

train_data, _ = YTCommentsDataset.split_train_test(3000, subset=0.8)
x_train, y_labels, y_test = ds.build_dataset(train_data, config=config)
```

Train the embedding model

```python
from david_sentiment import SentimentModel

ytc_sentiment = SentimentModel(config)
ytc_sentiment.train_model(x_train, y_labels)
```

Save the project: Call `save_project()` to create the project directories which saves all the essential settings for initiating a previous state, including; the trained-model and tokenizers vocab files:

- config file         : `<project_name>/config.init`
- trained model       : `<project_name>/model/model.h5`
- tokenizer vocab     : `<project_name>/vocab/vocab.pkl`

```python
ytc_sentiment.save_project()
```

Loading a saved project

```python
from david_sentiment import SentimentConfig, SentimentModel
config = SentimentConfig.load_project('ytc_sentiment/config.ini')
ytc_sentiment = SentimentModel(config)

print(ytc_sentiment)
'<SentimentModel(max_seqlen=62, vocab_shape=(2552, 100))>'
```

## Results

- With punctuation

```python
ytc_sentiment.print_predict("hello, world! i am glad this demo worked! :)")
  "input: hello, world! i am glad this demo worked! :) < pos(😍) (98.3824)% >"
```

- Without punctuation

```python
ytc_sentiment.print_predict("hello world I am glad this demo worked")
  "input: hello world I am glad this demo worked < pos(😀) (91.5674)% >"
```

**Textblob** vs ***SentimentModel*** trained on `1132` samples and `100` epochs.

```markdown
💬 (Textblob=0.0, SentimentModel=99.8896)
  😍 - pewdiepie plz u subcribe me and make a video on me

💬 (Textblob=0.0, SentimentModel=91.9985)
  😀 - You should make a video of you playing PUBG on this phone.

💬 (Textblob=0.0, SentimentModel=48.4672)
  😶 - If it's supposed to be an april fools

💬 (Textblob=0.0, SentimentModel=95.139)
  🤗 - Would you please make a video on Funcl W1 and Funcl AI earphones.

💬 (Textblob=0.0, SentimentModel=78.5567)
  😁 - Will you make a video on it ?

💬 (Textblob=0.0, SentimentModel=98.7835)
  😍 - Please think about it and make a video if you can.

💬 (Textblob=0.0, SentimentModel=94.1769)
  🤗 - we could hope to see in 2020??

💬 (Textblob=0.0, SentimentModel=98.7844)
  😍 - Make a video about not a smartphone plzzzzzzz

💬 (Textblob=0.0, SentimentModel=47.5426)
  😶 - Think about that.

💬 (Textblob=0.0, SentimentModel=98.4927)
  😍 - can you make a video on how to make thumbnail.

💬 (Textblob=0.0, SentimentModel=1.5344)
  🤬 - Please make a video about the vivo nex 2! 🙏

💬 (Textblob=0.0, SentimentModel=89.943)
  😀 - Your biggest fan

💬 (Textblob=0.0, SentimentModel=97.6116)
  😍 - Please make a video on how to use Facebook without internet.

💬 (Textblob=0.0, SentimentModel=61.0681)
  😑 - A BIG DEAL

💬 (Textblob=0.0, SentimentModel=91.3205)
  😀 - but I use my phone a lot for work and Netflix

💬 (Textblob=0.0, SentimentModel=40.8797)
  😒 - so why stop.

💬 (Textblob=0.0, SentimentModel=97.6973)
  😍 - Health, wealth and mind.

💬 (Textblob=0.0, SentimentModel=55.4884)
  😑 - Dose

💬 (Textblob=0.0, SentimentModel=42.6375)
  😒 - I would like to know your opinion.

💬 (Textblob=0.0, SentimentModel=26.5492)
  😤 - Liza don’t believe those hater lovers are here for you
```
