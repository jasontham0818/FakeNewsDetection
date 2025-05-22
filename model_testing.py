import os
import joblib
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import tensorflow as tf

# Load models and tokenizer paths
lr_model = joblib.load('models/lr/best_logistic_model.pkl')
vectorizer = joblib.load('models/lr/best_logistic_tokenizer.pkl')

best_rnn_model = tf.keras.models.load_model('models/rnn/best_rnn_model.keras')
rnn_tokenizer = joblib.load('models/rnn/best_rnn_tokenizer.pkl')

lstm_model = tf.keras.models.load_model('models/lstm/best_lstm_model.keras')
lstm_tokenizer = joblib.load('models/lstm/best_lstm_tokenizer.pkl')

# Load, preprocess, and clean testing data
testing_data = pd.read_csv("data/test_data.csv").dropna()

# Define stopwords and punctuation
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


def strip_html(text):
    if isinstance(text, float):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)


def remove_urls(text):
    return re.sub(r'http\S+', '', text)


def remove_stopwords(text):
    final_text = [i.strip() for i in text.split() if i.strip().lower() not in stop]
    return " ".join(final_text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_square_brackets(text)
    text = remove_urls(text)
    text = remove_stopwords(text)
    return text


testing_data["text"] = testing_data["text"].apply(denoise_text)

# Separate text and labels from the random sample
new_text = testing_data['text']
new_label = testing_data['label']

# Initialize results DataFrame with text and true label columns
results_df = pd.DataFrame({
    'text': new_text,
    'True_label': new_label
})

# Logistic Regression predictions
lr_predictions = lr_model.predict(vectorizer.transform(new_text))
results_df['LR_predicted_label'] = lr_predictions

# Best RNN predictions
new_text_pad = tf.keras.preprocessing.sequence.pad_sequences(
    rnn_tokenizer.texts_to_sequences(new_text), maxlen=500
)
best_rnn_predictions = (best_rnn_model.predict(new_text_pad)> 0.5).astype("int32").flatten()
results_df['Best_RNN_predicted_label'] = best_rnn_predictions

# LSTM predictions
lstm_text = tf.keras.preprocessing.sequence.pad_sequences(
    lstm_tokenizer.texts_to_sequences(new_text), maxlen=500
)
lstm_predictions = (lstm_model.predict(lstm_text) > 0.5).astype("int32").flatten()
results_df['lstm_prediction_label'] = lstm_predictions

# Save results to CSV
os.makedirs('result', exist_ok=True)
results_df.to_csv('result/result.csv', index=False)
print("Results saved")

# Calculate and display accuracies
lr_accuracy = (results_df['LR_predicted_label'] == new_label).mean()
best_rnn_accuracy = (results_df['Best_RNN_predicted_label'] == new_label).mean()
lstm_accuracy = (results_df['lstm_prediction_label'] == new_label).mean()

# Display Transformer model accuracies
print(f"Logistic Regression Model Accuracy: {lr_accuracy * 100:.2f}%")
print(f"Best RNN Model Accuracy: {best_rnn_accuracy * 100:.2f}%")
print(f"LSTM Model Accuracy: {lstm_accuracy * 100:.2f}%")
