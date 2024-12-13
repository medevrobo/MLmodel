import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_text TEXT NOT NULL,
                        user_feedback TEXT NOT NULL,
                        model_prediction TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_text TEXT NOT NULL,
                        label TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def save_feedback(input_text, user_feedback, model_prediction, date_submitted):
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO feedback (input_text, user_feedback, model_prediction, date_submitted)
        VALUES (?, ?, ?, ?)
        ''',
        (input_text, user_feedback, model_prediction, date_submitted)
    )
    conn.commit()
    conn.close()


# def save_feedback(input_text, user_feedback, model_prediction):
#     conn = sqlite3.connect('feedback.db')
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO feedback (input_text, user_feedback, model_prediction) VALUES (?, ?, ?)',
#                    (input_text, user_feedback, model_prediction))
#     conn.commit()
#     conn.close()

def get_feedback_data():
    conn = sqlite3.connect('feedback.db')
    feedback_data = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return feedback_data

def add_date_submitted_column():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE feedback ADD COLUMN date_submitted TEXT DEFAULT NULL")
    conn.commit()
    conn.close()

def backfill_date_submitted():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE feedback SET date_submitted = '2024-01-01' WHERE date_submitted IS NULL")
    conn.commit()
    conn.close()