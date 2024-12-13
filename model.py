# import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# import pandas as pd
# from utils import preprocess_text
# import sqlite3




# def train_initial_model():
#     # Load and preprocess initial training data
#     pass  # Include your initial training logic here

# def retrain_model():
#     conn = sqlite3.connect('feedback.db')
#     feedback_df = pd.read_sql_query("SELECT input_text, user_feedback FROM feedback", conn)
#     feedback_df['label'] = feedback_df['user_feedback'].map({"Positive": 1, "Neutral": 0, "Negative": -1})
#     training_df = pd.read_sql_query("SELECT input_text, label FROM training_data", conn)
#     conn.close()

#     combined_df = pd.concat([feedback_df[['input_text', 'label']], training_df], ignore_index=True)
#     combined_df['clean_text'] = combined_df['input_text'].apply(preprocess_text)
#     X = vectorizer.transform(combined_df['clean_text'])
#     y = combined_df['label']

#     model = LogisticRegression()
#     model.fit(X, y)
#     joblib.dump(model, "sentiment_model.pkl")

# def predict_sentiment(text, model, vectorizer):
#     clean_text = preprocess_text(text)
#     transformed_text = vectorizer.transform([clean_text])
#     prediction_code = model.predict(transformed_text)[0]
#     sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
#     return sentiment_map[prediction_code]


import joblib
import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils import preprocess_text


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

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Step 2: Preprocessing Function
def preprocess_text(text):
    # print("5")
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


    # Step 7: Deployment Example - Real-time Prediction
def predict_sentiment(text):
    clean_text = preprocess_text(text)
    transformed_text = vectorizer.transform([clean_text])
    prediction = model.predict(transformed_text)[0]
    sentiment = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return sentiment[prediction]

    print("11")

def evaluate_model():
    """
    Evaluate the model on the original test data and user feedback.
    """
    # Load test data and preprocess
    test_data = pd.read_csv("test_data.csv")
    # Drop rows with NaN in the 'text' column
    test_data = test_data.dropna(subset=['text'])
    # # Alternatively, fill NaN with an empty string
    # test_data['text'] = test_data['text'].fillna('')
    test_data['clean_text'] = test_data['text'].apply(preprocess_text)

    X_test = vectorizer.transform(test_data['clean_text'])
    y_test = test_data['label']

    # Predict and calculate accuracy on original test data
    y_pred_test = model.predict(X_test)
    test_data_accuracy = accuracy_score(y_test, y_pred_test)

    # Evaluate on user feedback
    conn = sqlite3.connect('feedback.db')
    feedback_df = pd.read_sql_query("SELECT input_text, user_feedback FROM feedback", conn)
    conn.close()

    # Preprocess feedback data
    feedback_df['clean_text'] = feedback_df['input_text'].apply(preprocess_text)
    feedback_df['label'] = feedback_df['user_feedback'].map({"Positive": 1, "Neutral": 0, "Negative": -1})
    feedback_df = feedback_df.dropna(subset=['label'])

    if not feedback_df.empty:
        X_feedback = vectorizer.transform(feedback_df['clean_text'])
        y_feedback = feedback_df['label']
        y_pred_feedback = model.predict(X_feedback)
        feedback_data_accuracy = accuracy_score(y_feedback, y_pred_feedback)
    else:
        feedback_data_accuracy = None

    return test_data_accuracy, feedback_data_accuracy


def train_initial_model():
    """
    Train the model with initial data.
    Include logic to preprocess, split data, and save the model/vectorizer.
    """
    # Load and preprocess your initial training data
    
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



    # Apply preprocessing
    data['clean_text'] = data['text'].apply(preprocess_text)

    # Step 3: Split Dataset
    X = data['clean_text']
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save test data for future evaluation
    test_data = pd.DataFrame({'text': X_test, 'label': y_test})
    test_data.to_csv("test_data.csv", index=False)

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



    # Test prediction
    example_text = "I love this airline! Their service is worst."
    print(f"Sentiment for '{example_text}':", predict_sentiment(example_text))


def retrain_model():
    """
    Retrain the model using feedback data and original training data.
    """
    # Load feedback data
    conn = sqlite3.connect('feedback.db')
    feedback_df = pd.read_sql_query("SELECT input_text, user_feedback FROM feedback", conn)

    # Map user feedback to numerical labels
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    feedback_df['label'] = feedback_df['user_feedback'].map(sentiment_map)

    # Drop rows with invalid labels
    feedback_df = feedback_df.dropna(subset=['label'])

    # Load original training data
    training_df = pd.read_sql_query("SELECT input_text, label FROM training_data", conn)
    conn.close()

    # Combine feedback and training data
    combined_df = pd.concat([feedback_df[['input_text', 'label']], training_df], ignore_index=True)

    # Preprocess and vectorize
    combined_df['clean_text'] = combined_df['input_text'].apply(preprocess_text)

    # Ensure labels are integers
    combined_df['label'] = combined_df['label'].astype(int)

    X = vectorizer.transform(combined_df['clean_text'])
    y = combined_df['label']

    # Retrain the model
    new_model = LogisticRegression()
    new_model.fit(X, y)

    # Save the updated model
    joblib.dump(new_model, "sentiment_model.pkl")
    print("Model retrained successfully.")

def retrain_model_if_needed():
    """
    Retrain the model if 10 or more new data points are available.
    """
    conn = sqlite3.connect('feedback.db')
    feedback_df = pd.read_sql_query("SELECT input_text, user_feedback FROM feedback", conn)
    conn.close()

    if len(feedback_df) >= 10:
        feedback_df['clean_text'] = feedback_df['input_text'].apply(preprocess_text)
        feedback_df['label'] = feedback_df['user_feedback'].map({"Positive": 1, "Neutral": 0, "Negative": -1})
        feedback_df = feedback_df.dropna(subset=['label'])

        # Load and preprocess original training data
        training_data = pd.read_csv("Tweets.csv")
        training_data['clean_text'] = training_data['text'].apply(preprocess_text)
        training_data['label'] = training_data['airline_sentiment'].map({"positive": 1, "neutral": 0, "negative": -1})

        # Combine datasets
        combined_data = pd.concat([training_data, feedback_df], ignore_index=True)

        # Vectorize and retrain
        X = vectorizer.fit_transform(combined_data['clean_text'])
        y = combined_data['label']

        new_model = LogisticRegression()
        new_model.fit(X, y)

        # Save updated model
        joblib.dump(new_model, "sentiment_model.pkl")
        print("Model retrained successfully.")
    else:
        print("Not enough new data to retrain.")
