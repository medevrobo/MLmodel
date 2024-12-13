# Sentiment Analysis with User Feedback Integration

This project implements a sentiment analysis application using Flask. It covers data collection, preprocessing, feature engineering, model training, evaluation, and deployment with a user feedback loop to improve the model over time.

## Table of Contents
1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Preprocessing](#preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Deployment](#deployment)
8. [Feedback Integration and Retraining](#feedback-integration-and-retraining)
9. [Dashboard and Monitoring](#dashboard-and-monitoring)
10. [How to Run](#how-to-run)

---

## Overview
The goal of this project is to classify text sentiments (positive, neutral, or negative) based on user feedback. The application supports real-time sentiment analysis, collects user feedback, and retrains the model with new data to improve its accuracy.

## Data Collection
- **Source**: Initial training data is sourced from the `Tweets.csv` dataset, which contains text data and sentiment labels.
- **Columns Used**:
  - `text`: The tweet text.
  - `airline_sentiment`: Sentiment labels (`positive`, `neutral`, `negative`).

## Preprocessing
Text preprocessing is performed to clean and normalize the data:
1. Remove URLs, mentions, and hashtags.
2. Remove special characters and convert text to lowercase.
3. Apply lemmatization using the `WordNetLemmatizer`.
4. Remove stop words using NLTK's English stop words list.

The cleaned text is saved as a new column, `clean_text`.

## Feature Engineering
- **Vectorization**: The text data is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
- **Parameters**:
  - `max_features=5000`: Limit the vocabulary size to the top 5000 terms.

## Model Training
### Initial Training
1. The preprocessed dataset is split into training and testing sets using an 80-20 split.
2. **Logistic Regression** is trained with a grid search over the following hyperparameters:
   - `C`: Regularization strength.
   - `penalty`: Regularization type (`l1`, `l2`).
   - `solver`: Optimization algorithm.
   - `max_iter`: Maximum iterations.
3. The best model is saved as `sentiment_model.pkl`, and the TF-IDF vectorizer is saved as `vectorizer.pkl`.

### Retraining
- The model is retrained using both the original training data and new data collected from user feedback.
- Incremental retraining is performed after collecting feedback for 10 new data points.

## Model Evaluation
- Evaluation is performed using:
  - **Classification Report**: Provides precision, recall, and F1 scores.
  - **Accuracy**: Computed on the test dataset and new user feedback data.
  - **Cross-Validation**: Assesses generalization performance using 5-fold cross-validation on training data.

Performance metrics are logged to monitor the model over time.

## Deployment
- A **Flask Web Application** provides a user interface for:
  - Submitting text for sentiment analysis.
  - Viewing predicted sentiment.
  - Providing feedback on predictions.
- It is also deployed on Render: https://mlmodel-j2zp.onrender.com

### Key Components
1. **Home Page**: Text input form for real-time sentiment analysis.
2. **Dashboard**: Displays:
   - Model performance metrics over time.
   - Accuracy trends for original test data and feedback data.

## Feedback Integration and Retraining
1. User feedback is collected via a form.
2. Feedback is logged into an SQLite database with columns:
   - `input_text`: The original user input.
   - `user_feedback`: Correct sentiment provided by the user.
3. When 10 new feedback points are collected, the model is retrained.

## Dashboard and Monitoring
- **Features**:
  - Visualize accuracy trends over time for test and feedback datasets.
  - Summarize feedback data statistics.
- **Technologies**: Python's `matplotlib` and `flask` are used for visualization and interaction.

## How to Run
### Prerequisites
- Python 3.7+
- Required libraries: Install using `pip install -r requirements.txt`.

### Steps
1. Clone the repository.
2. Train the initial model:
   ```bash
   python model.py
   ```
3. Start the Flask application:
   ```bash
   python app.py
   ```
4. Access the application in a web browser at `http://127.0.0.1:8001`.
5. Interact with the application:
   - Input text for sentiment prediction.
   - Provide feedback for incorrect predictions.
   - View the dashboard for performance monitoring.

---

## File Structure
```
.
├── app.py              # Flask application
├── model.py            # Model training and retraining logic
├── templates/
│   ├── index.html      # Home page UI
│   ├── dashboard.html  # Dashboard UI
├── static/             # Static assets (CSS, JS)
├── Tweets.csv          # Initial dataset
├── feedback.db         # SQLite database for feedback
├── sentiment_model.pkl # Trained model
├── vectorizer.pkl      # TF-IDF vectorizer
└── requirements.txt    # Required Python libraries
```

