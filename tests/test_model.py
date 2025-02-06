import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TextClassificationPipeline,
)


def test_model_accuracy():
    # Load data
    data = pd.read_csv("data/sample_reviews.csv")

    # Split data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Load model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Initialize sentiment analysis pipeline
    sentiment_model = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, return_all_scores=False
    )

    # Function to get sentiment from review
    def get_sentiment(review):
        result = sentiment_model(review)[0]
        return result["label"].lower()

    # Apply the model to the test set
    test_data["predicted_sentiment"] = test_data["review"].apply(get_sentiment)

    # Print each review with its predicted and actual sentiment
    for index, row in test_data.iterrows():
        print(f"Review: {row['review']}")
        print(f"Predicted Sentiment: {row['predicted_sentiment']}")
        print(f"Actual Sentiment: {row['sentiment']}\n")

    # Evaluate the model
    accuracy = accuracy_score(test_data["sentiment"], test_data["predicted_sentiment"])

    # Assert that the accuracy is above a certain threshold
    assert accuracy > 0.6, f"Model accuracy is too low: {accuracy}"
