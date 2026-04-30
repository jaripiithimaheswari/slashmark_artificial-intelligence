# Import libraries
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (make sure CSV has 'text' and 'sentiment' columns)
script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "reviews.csv"
if not csv_path.exists():
    raise FileNotFoundError(
        f"Could not find dataset at {csv_path}. Place reviews.csv next to sentiment.py or update the path."
    )

data = pd.read_csv(csv_path)

# Initialize stopwords
stop_words = set(ENGLISH_STOP_WORDS)

# Text preprocessing function
def preprocess(text):
    text = str(text).lower()  # convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove special characters
    words = text.split()  # tokenize
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

# Apply preprocessing
data['clean_text'] = data['text'].apply(preprocess)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['sentiment']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with new input
try:
    while True:
        user_input = input("\nEnter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        cleaned = preprocess(user_input)
        vector = tfidf.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0]
        
        print(f"\nYour input: '{user_input}'")
        print(f"Predicted Sentiment: {prediction}")
        print("\nConfidence Scores:")
        for sentiment, score in zip(model.classes_, confidence):
            print(f"  {sentiment}: {score:.4f} ({score*100:.2f}%)")
except (KeyboardInterrupt, EOFError):
    print("\nExiting sentiment tester.")
