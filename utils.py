import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = re.sub(r'http\\S+|www\\S+', '', text)  # Remove URLs
    text = re.sub(r'@\\w+|#\\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\\s]', '', text)  # Remove special characters
    text = text.lower()  # Lowercase
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text
