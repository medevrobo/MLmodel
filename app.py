from flask import Flask, render_template, request, redirect, jsonify
from database import init_db, save_feedback, get_feedback_data, add_date_submitted_column, backfill_date_submitted
from model import retrain_model, predict_sentiment, train_initial_model, evaluate_model, retrain_model_if_needed
import joblib
import sqlite3
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    user_input = None
    if request.method == 'POST':
        if 'user_input' in request.form:
            user_input = request.form['user_input']
            prediction = predict_sentiment(user_input)
        elif 'feedback' in request.form:
            user_input = request.form.get('hidden_user_input')
            feedback = request.form.get('feedback')
            model_prediction = request.form.get('hidden_model_prediction')
            if user_input is None or feedback is None or model_prediction is None:
                print("Missing form data:", request.form)  # Debugging log
                return "Error: Missing form data. Please try again.", 400
            
            # Save feedback with timestamp
            date_submitted = datetime.now().strftime("%Y-%m-%d")  # Current date
            save_feedback(user_input, feedback, model_prediction, date_submitted)

            # Retrain the model if needed
            retrain_model_if_needed()

    return render_template('index.html', prediction=prediction, user_input=user_input)


@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('feedback.db')
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()

    # Aggregate sentiment trends
    sentiment_counts = df['user_feedback'].value_counts().to_dict()
    total_feedback = len(df)

    print("+++++++++++++++++++")
    print(df)
    print("++++++++++++++++++++")

    # Calculate model performance over time
    df['correct'] = df['user_feedback'] == df['model_prediction']
    accuracy_over_time = df['correct'].mean() * 100  # Overall accuracy percentage

    # Evaluate model accuracy on test data and feedback data
    test_data_accuracy, feedback_data_accuracy = evaluate_model()

    # Calculate sentiment trends over time
    trends = defaultdict(lambda: {"Positive": 0, "Neutral": 0, "Negative": 0})
    for _, row in df.iterrows():
        date = row['date_submitted']
        sentiment = row['user_feedback']
        trends[date][sentiment] += 1

    trend_data = [{"date": date, **counts} for date, counts in trends.items()]

    return render_template(
        'dashboard.html',
        sentiment_counts=sentiment_counts,
        total_feedback=total_feedback,
        accuracy_over_time=accuracy_over_time,
        test_data_accuracy=test_data_accuracy * 100,
        feedback_data_accuracy=(feedback_data_accuracy * 100 if feedback_data_accuracy else "No data"),
        trend_data=trend_data,
    )


@app.route('/sentiment_trends')
def sentiment_trends():
    """API endpoint for sentiment trends data."""
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()

    # Query feedback data with dates
    query = """
        SELECT date_submitted, user_feedback 
        FROM feedback
        WHERE user_feedback IS NOT NULL
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    # Process data into trends
    trends = defaultdict(lambda: {"Positive": 0, "Neutral": 0, "Negative": 0})
    for row in rows:
        date = row[0]
        sentiment = row[1]
        trends[date][sentiment] += 1

    trend_data = [{"date": date, **counts} for date, counts in trends.items()]
    return jsonify(trend_data)


if __name__ == '__main__':
    # Initialize the database
    init_db()

    # # Add the date_submitted column if it doesn't exist
    # add_date_submitted_column()

    # # Backfill the date_submitted column for existing data
    # backfill_date_submitted()

    # Start the Flask app
    app.run(debug=True, port=8001)


# --------------------------


# from flask import Flask, render_template, request, redirect, url_for
# from database import init_db, save_feedback, get_feedback_data
# from model import retrain_model, predict_sentiment, train_initial_model, evaluate_model, retrain_model_if_needed
# import joblib
# import sqlite3
# import pandas as pd

# # Initialize Flask app
# app = Flask(__name__)

# # Load the model and vectorizer
# model = joblib.load("sentiment_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     prediction = None
#     user_input = None
#     if request.method == 'POST':
#         if 'user_input' in request.form:
#             user_input = request.form['user_input']
#             prediction = predict_sentiment(user_input)
#         elif 'feedback' in request.form:
#             user_input = request.form.get('hidden_user_input')
#             feedback = request.form.get('feedback')
#             model_prediction = request.form.get('hidden_model_prediction')
#             if user_input is None or feedback is None or model_prediction is None:
#                 print("Missing form data:", request.form)  # Debugging log
#                 return "Error: Missing form data. Please try again.", 400
#             save_feedback(user_input, feedback, model_prediction)
#             retrain_model()
#     return render_template('index.html', prediction=prediction, user_input=user_input)


# @app.route('/dashboard')
# def dashboard():
#     conn = sqlite3.connect('feedback.db')
#     df = pd.read_sql_query("SELECT * FROM feedback", conn)
#     conn.close()

#     # Aggregate sentiment trends
#     sentiment_counts = df['user_feedback'].value_counts().to_dict()
#     total_feedback = len(df)

#     print("************")
#     print(df)
#     print("+++++++++++++")

#     # Calculate model performance over time
#     df['correct'] = df['user_feedback'] == df['model_prediction']
#     accuracy_over_time = df['correct'].mean() * 100  # Overall accuracy percentage

#     test_data_accuracy, feedback_data_accuracy = evaluate_model()
#     retrain_model_if_needed()

#     return render_template('dashboard.html', sentiment_counts=sentiment_counts, total_feedback=total_feedback, accuracy_over_time=accuracy_over_time, 
#                         test_data_accuracy=test_data_accuracy * 100,
#                         feedback_data_accuracy=(feedback_data_accuracy * 100 if feedback_data_accuracy else "No data")
#                         )


# if __name__ == '__main__':
#     init_db()
#     app.run(debug=True, port=8001)
