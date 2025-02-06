import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TextClassificationPipeline,
)

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

# Evaluate the model
accuracy = accuracy_score(test_data["sentiment"], test_data["predicted_sentiment"])
report = classification_report(test_data["sentiment"], test_data["predicted_sentiment"])
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")


# Print sentences with their actual and predicted sentiments
for index, row in test_data.iterrows():
    print(
        f"Review: {row['review']}\nActual Sentiment: {row['sentiment']}\nPredicted Sentiment: {row['predicted_sentiment']}\n"
    )
