import david_sentiment.dataset as ds
from david_sentiment import YTCSentimentConfig, YTCSentimentModel
from david_sentiment.utils import (test_polarity_attention_weights,
                                   test_unseen_samples)

config = YTCSentimentConfig(
    project_dir="model-001",
    min_strlen=20,
    max_strlen=10000,
    mintoken_freq=1,
    reduce_length=True,
    remove_urls=True,
)

batch = ds.BatchDB(
    [
        ds.Fetch("unbox", "%make a video%"),
        ds.Fetch("v1", "%make a video%"),
        ds.Fetch("v2", "%make a video%"),
    ]
)

x_texts, x_labels, y_tests = ds.fit_batch_to_dataset(batch, config)
ytc_sentiment = YTCSentimentModel(config)
ytc_sentiment.train_model(x_texts, x_labels)


# test the model with some demo functions
print("\ntesting the model...")
test_polarity_attention_weights(ytc_sentiment)
test_unseen_samples(ytc_sentiment, y_tests, print_k=50)

# save the model
print("\nsaving the model...")
ytc_sentiment.save_project(exist_ok=True)
