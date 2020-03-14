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

# you can define everything upfront or as you go.
config = SentimentConfig(project_dir="my-model",
                         max_strlen=1000,
                         min_strlen=20,
                         epochs=13,
                         enforce_ascii=True,
                         remove_urls=True,
                         glove_ndim="300d",)  
```

Build a dataset from database queries:

```python
import david_sentiment.dataset as ds

batch = ds.BatchDB([ds.Fetch('unbox', "%make a video%"),
                    ds.Fetch('v1', "%make a video%"),])
trainable = ds.build_dataset(batch, config, untrainable=False) # default
```

- The `build_dataset()` method works for any iterable of strings.
  - A `trainable` is simply a document annotated by the meta learner (In this case TextBlob).

> At the moment only binary classification datasets are compatible with the full pipeline. But I am planning  to implement multi-categorical features for the same semantic tasks.

```python
from david_sentiment.dataset import YouTubeComments
from david_sentiment.dataset import build_dataset

comments = YouTubeComments()
dataset = dataset.texts() # returns a generator
trainable, untrainable = build_dataset(dataset, config, untrainable=True)
```

- A document goes through a carefully designed pipeline; It is important for you to know what is going on in the background. So if you want to inspect why your documents where `Untrainable` - you can simple ask the method to return it. Here's the output and information after executing the `build_dataset()` method.

```bash
...
⚠  normalizing 61478 string sequences from batch.
sequences: 100%|██████████| 61478/61478 [00:54<00:00, 1130.32/s]

✔  removed 594 strings of length >= 1000.
ℹ  returning batch with 60884 samples.
⚠  tokenizing strings to sentences, spaCy model: en_core_web_sm.
✔  removed 51264 strings of length <= 20.
ℹ  previous-size: 60884, new-size: 142342.
⚠  TextBlob annotating dataset with binary classes: 0 | 1
✔  annotation summary 🤖
ℹ  trainable: 73356, un-trainable: 68986
```

You can train the model with one line - or `step-by-step` **(see below)**

```python
from david_sentiment import SentimentModel

sentiment = SentimentModel(config)
sentiment.train(trainable) # List[Tuple[List[str], List[int]]]]
```

## Working with the model step-by-step (made easy)

The `SentimentModel` class holds all the essential properties of your dataset, like your vocabulary and all common attributes required for building, batching, compiling etc so you can focus on building, training, and experimenting.

- Also, why do we need a class for passing objects around?
  - Instead of passing a bunch of globals all over, you can keep everything in one place. Your workspace stays tidy so you can focus on training and building the model and not managing or trying to locate all those global variables.

- I have a lot of documents?
  - If you have over 5000K samples, I recommend dropping tokens with a frequency of 2 or more (but it depends on your dataset).

```python
# You can also add existing models and or any models you make later.
# this is a clean way to avoid collisions if you where to train many models with the same instance.
pt_model = SentimentModel.clone(sentiment, model)

# finally, transform the trainable document and its binary labels
# to the format the model expects (segment=True fits the document to 1:1 ratio)
# 1:1 meaning 50% 50% distribution on the [0, 1] binary classes (important!)
x_train, y_train, x_test, y_test = sentiment.transform(trainable, mincount=2)
len(x_train), len(x_test) 
...
'ℹ * Removed 13802 tokens from 26948'
(37744, 9436)
```

Getting the embedding layer for the model. `[50d, 100d, 200d, 300d]` available.

```python
embedding_layer = sentiment.embedding(l2=1e-6, ndim="300d")
...
✔ '<(dim=300, vocab=13147)>'
✔ 'embedding vocabulary 👻'
✔ 'Glove embeddings loaded from path:'
'/home/<usr>/david_models/glove/glove.6B/glove.6B.300d.txt'
```

```python
pt_model = sentiment.compile_net(None, layer=embedding, mode="pre-trained")
pt_model.summary()
```

```bash
...
Model: "david-sentiment (PT)"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 125, 300)          3944100   
_________________________________________________________________
flatten_1 (Flatten)          (None, 37500)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1200032   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 5,144,165
Trainable params: 1,200,065
Non-trainable params: 3,944,100
_________________________________________________________________
```

- And finally train your model!

```python
history = model.fit(x_train, y_train,
                    epochs=13,
                    batch_size=512,
                    validation_data=(x_test, y_test))
...
```

```bash
Train on 37744 samples, validate on 9436 samples
Epoch 1/13
37744/37744 [==============================] - 9s 249us/step - loss: 0.6003 - acc: 0.6701 - val_loss: 0.5387 - val_acc: 0.7174
Epoch 2/13
37744/37744 [==============================] - 9s 241us/step - loss: 0.4914 - acc: 0.7583 - val_loss: 0.5187 - val_acc: 0.7364
...
Epoch 13/13
37744/37744 [==============================] - 9s 249us/step - loss: 0.1589 - acc: 0.9447 - val_loss: 0.6189 - val_acc: 0.7568
```

## Plotting

```python
from david_sentiment.utils import plot_accuracy, plot_losses

plot_losses(history, show=True, save=False)
```

<img src="src/loss.png" />

```python
plot_accuracy(history, show=True, save=False)
```

<img src="src/acc.png" />

## Results

The results below are after training the model with `37744` samples, `13` epochs, and `200 dimensional` GloVe embeddings

- Network: (2-Dense Layers):
  - 1st layer = `32` *hidden-units*, activation = ***relu***
  - 2nd layer = `1` *hidden-unit*, activation = ***sigmoid***
  
- The last layer being the output scalar prediction regarding the sentiment of the input.

> **..** `+` **:)**

```python
sentiment.predict_print('idk how i feel anymore.. :)')
...
input: < pos(😊) (68.9933)% >
```

> **..** `+` **:(**

```python
sentiment.predict_print('idk how i feel anymore.. :(')
...
input: < neg(😡) (10.4026)% >
```

> **..**

```python
sentiment.predict_print('idk how i feel anymore..')
...
input: < pos(😶) (50.6473)% >
```

> **.**

```python
sentiment.predict_print("idk how i feel anymore.")
...
input: < neg(😬) (36.1655)% >
```

```python
from david_sentiment.utils import test_unseen_samples

# try running predictions on the un-trainable dataset (shuffled)
test_unseen_samples(sentiment, untrainable, k=13)
```

```markdown
💬 <old=0.0, new=96.3195, label=1>
 🤗 - I'm Japanese student , and I think this video is valuable to study how to use python.

💬 <old=0.0, new=45.5757, label=0>
 😒 - But I'm also trying to keep up with my fitness.

💬 <old=0.0, new=48.2212, label=0>
 😶 - I'll see what I can do
 
💬 <old=0.0, new=92.5349, label=1>
 🤗 - You are actually make me realize the importance of focus on Myself instead of criticizing others.

💬 <old=0.0, new=87.7247, label=1>
 😀 - You just want some likes.
 
 💬 <old=0.0, new=26.2294, label=0>
 😤 - Have ever thought that the rings for the warnings are to notify us if we forget

💬 <old=0.0, new=6.0083, label=0>
 🤬 - “...50 lbs of 💩 in a 5 lb bag!”

💬 <old=0.0, new=97.5032, label=1>
 😍 - I wanted a reliable phone with me until the Note 10 comes out.

💬 <old=0.0, new=91.0801, label=1>
 😀 - " I just can't get enough of it.

💬 <old=0.0, new=72.4068, label=1>
 😊 - but how i improve myself like python develper.

💬 <old=0.0, new=99.0795, label=1>
 😍 - and I think I need to master reading to master the meditation
 
 💬 <old=0.0, new=8.3481, label=0>
 😡 - Windows also has issues with Python versions above 3.4 so try to get 3.4.
 
 💬 <old=0.0, new=18.5559, label=0>
 😠 - only me with this problem???
```

## Saving/Loading

Save the project: Call `save_project()` to create the project directories which saves all the essential settings for initiating a previous state, including; the trained-model and tokenizers vocab files:

- project:
  - config file         : `<project_name>/config.init`
  - trained model       : `<project_name>/model/model.h5`
  - tokenizer vocab     : `<project_name>/vocab/vocab.pkl`

> saving

```python
sentiment.save_project()
```

> loading

```python
from david_sentiment import SentimentConfig, SentimentModel

config = SentimentConfig.load_project('my-model/config.ini')
sentiment = SentimentModel(config)
print(sentiment)
'<SentimentModel(max_seqlen=62, vocab_size=(2552, 100))>'
```

The parameters, model, and tokenizer are loaded automatically after passing the config object to the SentimentModel class.
