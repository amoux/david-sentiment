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

## Results

- with punctuation.

```python
ytc_sentiment.print_predict("hello, world! i am glad this demo worked! :)")
...
input: "hello, world! i am glad this demo worked! :)" < pos(😍) (98.3824)% >
```

- without punctuation.

```python
ytc_sentiment.print_predict("hello world I am glad this demo worked")
...
input: "hello world I am glad this demo worked" < pos(😀) (91.5674)% >
```

- `Textblob` vs `YTCSentimentModel` trained on `1132` samples and `100` epochs.

```markdown
  💬 (Textblob=0.0, YTCSentimentModel=99.8896)
    😍 - pewdiepie plz u subcribe me and make a video on me
  
  💬 (Textblob=0.0, YTCSentimentModel=91.9985)
    😀 - You should make a video of you playing PUBG on this phone.
  
  💬 (Textblob=0.0, YTCSentimentModel=48.4672)
    😶 - If it's supposed to be an april fools
  
  💬 (Textblob=0.0, YTCSentimentModel=98.5463)
    😍 - Plz make a video on India, cost & religion system....
  
  💬 (Textblob=0.0, YTCSentimentModel=52.7096)
    😑 - Plz make a video of redmi note 7 pro
  
  💬 (Textblob=0.0, YTCSentimentModel=31.7184)
    😳 - Go watch it.
  
  💬 (Textblob=0.0, YTCSentimentModel=79.7811)
    😁 - make a video covering each accessory please man!
  
  💬 (Textblob=0.0, YTCSentimentModel=95.4761)
    🤗 - Make a video on Redmi k20 pro
  
  💬 (Textblob=0.0, YTCSentimentModel=86.7035)
    😀 - It's about balance.
  
  💬 (Textblob=0.0, YTCSentimentModel=95.139)
    🤗 - Would you please make a video on Funcl W1 and Funcl AI earphones.
  
  💬 (Textblob=0.0, YTCSentimentModel=78.5567)
    😁 - Will you make a video on it ?
  
  💬 (Textblob=0.0, YTCSentimentModel=98.7835)
    😍 - Please think about it and make a video if you can.
  
  💬 (Textblob=0.0, YTCSentimentModel=94.1769)
    🤗 - we could hope to see in 2020??
  
  💬 (Textblob=0.0, YTCSentimentModel=98.7844)
    😍 - Make a video about not a smartphone plzzzzzzz
  
  💬 (Textblob=0.0, YTCSentimentModel=96.7084)
    🤗 - You don’t have to be a bitch.
  
  💬 (Textblob=0.0, YTCSentimentModel=47.5426)
    😶 - Think about that.
  
  💬 (Textblob=0.0, YTCSentimentModel=98.4927)
    😍 - can you make a video on how to make thumbnail.
  
  💬 (Textblob=0.0, YTCSentimentModel=1.5344)
    🤬 - Please make a video about the vivo nex 2! 🙏
  
  💬 (Textblob=0.0, YTCSentimentModel=89.943)
    😀 - Your biggest fan
  
  💬 (Textblob=0.0, YTCSentimentModel=97.6116)
    😍 - Please make a video on how to use Facebook without internet.
  
  💬 (Textblob=0.0, YTCSentimentModel=61.0681)
    😑 - A BIG DEAL
  
  💬 (Textblob=0.0, YTCSentimentModel=91.3205)
    😀 - but I use my phone a lot for work and Netflix
  
  💬 (Textblob=0.0, YTCSentimentModel=40.8797)
    😒 - so why stop.
  
  💬 (Textblob=0.0, YTCSentimentModel=97.6973)
    😍 - Health, wealth and mind.
  
  💬 (Textblob=0.0, YTCSentimentModel=55.4884)
    😑 - Dose
  
  💬 (Textblob=0.0, YTCSentimentModel=42.6375)
    😒 - I would like to know your opinion.
  
  💬 (Textblob=0.0, YTCSentimentModel=26.5492)
    😤 - Liza don’t believe those hater lovers are here for you
```
