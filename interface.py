import tkinter as tk
from tkinter import messagebox
import joblib
import re
import string
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


# Function to flag uncertain predictions (probability close to 0.5)
def flag_uncertain(prob):
    # Flagging 'Partially True' if probability is between 0.4 and 0.6
    if 0.4 < prob < 0.6:
        return 'Partially True'
    elif prob >= 0.6:
        return 'Fake'  # Label 1
    else:
        return 'True'  # Label 0


# Function to implement weighted voting
def weighted_voting(predictions, weights):
    # Check if any model suggests "Partially True"
    if 'Partially True' in predictions:
        return 'Partially True'

    # Otherwise, apply weighted voting based on scores
    prediction_scores = {'Fake': 0, 'True': 0}

    for i, pred in enumerate(predictions):
        if pred == 'Fake':
            prediction_scores['Fake'] += weights[i]
        elif pred == 'True':
            prediction_scores['True'] += weights[i]

    # Return the class with the highest score (excluding Partially True as it's already handled)
    return max(prediction_scores, key=prediction_scores.get)


# Define the prediction function
def predict_news():
    text = text_input.get("1.0", "end-1c")  # Get text from the input box

    if text == "":
        messagebox.showwarning("Input Error", "Please enter some text to classify.")
        return

    # Check if the input is purely numeric (contains only digits)
    if text.strip().isdigit():  # Use strip() to ignore leading/trailing spaces
        messagebox.showwarning("Input Error", "Please enter text only.")
        return

    # Denoise the text
    cleaned_text = denoise_text(text)

    # Logistic Regression prediction (probability)
    lr_prob = lr_model.predict_proba(vectorizer.transform([cleaned_text]))[:, 1][0]
    lr_prediction = flag_uncertain(lr_prob)

    # Best RNN prediction (probability)
    rnn_text_pad = tf.keras.preprocessing.sequence.pad_sequences(
        rnn_tokenizer.texts_to_sequences([cleaned_text]), maxlen=500
    )
    rnn_prob = best_rnn_model.predict(rnn_text_pad)[0]
    rnn_prediction = flag_uncertain(rnn_prob)

    # LSTM prediction (probability)
    lstm_text = tf.keras.preprocessing.sequence.pad_sequences(
        lstm_tokenizer.texts_to_sequences([cleaned_text]), maxlen=500
    )
    lstm_prob = lstm_model.predict(lstm_text)[0]
    lstm_prediction = flag_uncertain(lstm_prob)

    # Combine all model predictions into a list
    predictions = [lr_prediction, rnn_prediction, lstm_prediction]

    # Weights for models based on their trustworthiness or performance (adjust as needed)
    model_weights = [0.98, 0.973, 1.0]

    # Use weighted voting
    final_prediction = weighted_voting(predictions, model_weights)

    # Display results
    result_text = f"Logistic Regression Prediction: {lr_prediction}\n"
    result_text += f"RNN Prediction: {rnn_prediction}\n"
    result_text += f"LSTM Prediction: {lstm_prediction}\n"
    result_text += f"\nFinal Prediction: {final_prediction}"

    result_label.config(text=result_text)


# Create the GUI
root = tk.Tk()
root.title("Fake News Detection")

# Set the window size
root.geometry("600x500")  # Adjust the size here

# Create text input field
text_input_label = tk.Label(root, text="Enter News Text:", font=("Arial", 14))
text_input_label.pack(pady=5)

text_input = tk.Text(root, height=15, width=60, font=("Arial", 12))  # Larger text box
text_input.pack(pady=10)

# Create button to predict
predict_button = tk.Button(root, text="Predict", command=predict_news, font=("Arial", 14), width=20)
predict_button.pack(pady=15)

# Create result display label
result_label = tk.Label(root, text="", justify="left", font=("Arial", 12))
result_label.pack(pady=10)

# Run the app
root.mainloop()

