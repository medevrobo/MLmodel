# print("0")

# # Import libraries
# import pandas as pd
# import numpy as np
# import re
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
# import joblib  # For saving the model

# # Install dependencies if not already installed
# # !pip install nltk scikit-learn joblib

# # Download required NLTK data

# print("1")
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

# print("2")

# # Step 1: Load Dataset
# # Replace with your dataset path or API integration
# data = pd.read_csv("Tweets.csv")  # Example dataset
# data = data[['text', 'airline_sentiment']]  # Selecting necessary columns

# print("3")

# # Rename columns
# data.columns = ['text', 'sentiment']

# # Map sentiments to numerical values
# sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
# data['sentiment'] = data['sentiment'].map(sentiment_map)

# print("4")

# # Step 2: Preprocessing Function
# def preprocess_text(text):
#     text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
#     text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
#     text = text.lower()  # Lowercase
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))
#     text = " ".join(
#         [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
#     )
#     return text

# print("5")

# # Apply preprocessing
# data['clean_text'] = data['text'].apply(preprocess_text)

# # Step 3: Split Dataset
# X = data['clean_text']
# y = data['sentiment']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("6")

# # Step 4: Feature Engineering - TF-IDF Vectorization
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# print("7")

# # Step 5: Model Training - Logistic Regression
# model = LogisticRegression()
# model.fit(X_train_tfidf, y_train)

# print("8")

# # Step 6: Model Evaluation
# y_pred = model.predict(X_test_tfidf)
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))

# print("9")

# # Save the model and vectorizer
# joblib.dump(model, "sentiment_model.pkl")
# joblib.dump(vectorizer, "vectorizer.pkl")

# print("10")

# # Step 7: Deployment Example - Real-time Prediction
# def predict_sentiment(text):
#     clean_text = preprocess_text(text)
#     transformed_text = vectorizer.transform([clean_text])
#     prediction = model.predict(transformed_text)[0]
#     sentiment = {1: "Positive", 0: "Neutral", -1: "Negative"}
#     return sentiment[prediction]

# print("11")

# # Test prediction
# example_text = "I love this airline! Their service is worst."
# print(f"Sentiment for '{example_text}':", predict_sentiment(example_text))
# print(model.get_params())

#----------

print("0")

# Import libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving the model

print("1")

# Install dependencies if not already installed
# !pip install nltk scikit-learn joblib

# Download required NLTK data
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

print("2")

# Step 1: Load Dataset
data = pd.read_csv("Tweets.csv")  # Example dataset
data = data[['text', 'airline_sentiment']]  # Selecting necessary columns

print("3")

# Rename columns
data.columns = ['text', 'sentiment']

# Map sentiments to numerical values
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
data['sentiment'] = data['sentiment'].map(sentiment_map)

print("4")

# Step 2: Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Lowercase
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = " ".join(
        [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    )
    return text

print("5")

# Apply preprocessing
data['clean_text'] = data['text'].apply(preprocess_text)

# Step 3: Split Dataset
X = data['clean_text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("6")

# Step 4: Feature Engineering - TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("7")

# Step 5: Model Training - Logistic Regression with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200, 300]
}

model = LogisticRegression()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_

print("8")

# Step 6: Model Evaluation
y_pred = best_model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print("Cross-validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

print("9")

# Save the model and vectorizer
joblib.dump(best_model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("10")

# Step 7: Deployment Example - Real-time Prediction
def predict_sentiment(text):
    clean_text = preprocess_text(text)
    transformed_text = vectorizer.transform([clean_text])
    prediction = best_model.predict(transformed_text)[0]
    sentiment = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return sentiment[prediction]

print("11")

# Test prediction
example_text = "I love this airline! Their service is worst."
print(f"Sentiment for '{example_text}':", predict_sentiment(example_text))
