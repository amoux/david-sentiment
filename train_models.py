import os

from david.io import DataIO
from david_sentiment import SentimentConfig, SentimentModel
from david_sentiment.dataset import YouTubeComments, build_dataset
from david_sentiment.utils import plot_accuracy, plot_losses
from keras import layers
from keras.models import Sequential

data_file = "yt-comments"
config = SentimentConfig(min_strlen=30, max_strlen=3000)
sm = SentimentModel(config)

# build the dataset
train, test = [], []
if os.path.isfile(os.path.join("data", data_file)):
    train, test = DataIO.load_data(data_file)
else:
    comments = YouTubeComments()
    train, test = build_dataset(comments.texts(), config, True)
    prep_datasets = (train, test)
    DataIO.save_data(data_file, prep_datasets)

x_train, y_train, x_test, y_test = sm.transform(train, mincount=1)


def pretrained_model():
    # Model
    epochs = 10
    batch_size = 512

    embedding = sm.embedding(module="42b", ndim="300d")
    pt_model = sm.compile_net(None, layer=embedding, mode="pre-trained")
    history = pt_model.fit(x_train, y_train,
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=(x_test, y_test))

    plot_accuracy(history, show=False, save=True, name="acc-model-1.png")
    plot_losses(history, show=False, save=True, name="loss-model-1.png")

    pt = SentimentModel.clone(sm, pt_model)
    pt.epochs = epochs
    pt.project = "pt-model"
    pt.save_project(exist_ok=True)


def cnn_lstm_model():
    # Embedding
    vocab_size, ndim, max_seqlen = sm.input_shape
    embedd_name = f"David-{str(vocab_size)[:2]}K-{ndim}d"

    # Convolution
    filters = 64
    pool_size = 4
    kernel_size = 5
    padding = "valid"

    # LSTM
    lstm_output_size = 100

    # Model
    epochs = 5
    batch_size = 30

    model = Sequential(name="Sentiment (CNN-LSTM)")
    model.add(layers.Embedding(name=embedd_name,
                               input_dim=vocab_size,
                               output_dim=ndim,
                               input_length=max_seqlen,))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv1D(filters,
                            kernel_size,
                            padding=padding,
                            activation="relu",
                            strides=1))
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    model.add(layers.LSTM(lstm_output_size))
    model.add(layers.Dense(1))
    model.add(layers.Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test))

    plot_accuracy(history, save=True, show=False, name="acc-model-2.png")
    plot_losses(history, save=True, show=False, name="loss-model-2.png")

    cnn_lstm_model = SentimentModel.clone(sm, model)
    cnn_lstm_model.epochs = epochs
    cnn_lstm_model.project = "cnn-lstm-model"
    cnn_lstm_model.save_project(exist_ok=True)


if __name__ == '__main__':
    pretrained_model()
    cnn_lstm_model()
