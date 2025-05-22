import time
import joblib
import random
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.layers import LSTM
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

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


# Load data and apply denoising
train_data = pd.read_csv("data/train_data.csv")
valid_data = pd.read_csv("data/valid_data.csv")

train_data['text'] = train_data['text'].apply(denoise_text)
valid_data['text'] = valid_data['text'].apply(denoise_text)


def prepare_data():
    """Preprocess, shuffle, and split the data, and store it in global variables."""
    global x_train, x_valid, y_train, y_valid
    # Prepare data using train_data and valid_data directly
    x_train = train_data["text"]
    y_train = train_data["label"]
    x_valid = valid_data["text"]
    y_valid = valid_data["label"]

    # Shuffle the training data
    train_data_shuffled = train_data.sample(frac=1, random_state=SEED)  # Shuffle the entire train_data dataframe
    x_train = train_data_shuffled["text"]
    y_train = train_data_shuffled["label"]


def logistic_regression_model():
    """Train and evaluate a Logistic Regression model with TF-IDF."""
    start_time = time.time()

    # Initialize the TF-IDF vectorizer and fit it on the training data
    vectorizer = TfidfVectorizer(max_features=5000)
    x_tfidf_train = vectorizer.fit_transform(x_train)  # Fit on the training data
    y_train_data = y_train  # Use y_train after shuffle

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(x_tfidf_train, y_train_data)

    # Evaluate the model on validation data
    x_tfidf_valid = vectorizer.transform(x_valid)  # Transform the validation data
    y_valid_data = y_valid  # Validation labels

    y_pred_valid = model.predict(x_tfidf_valid)  # Predict on validation data
    f1_valid = f1_score(y_valid_data, y_pred_valid, average='binary')

    # Calculate confusion matrix for validation data
    cm_valid = confusion_matrix(y_valid_data, y_pred_valid)
    plot_confusion_matrix(cm_valid, 'Logistic Regression - Validation')

    # Save the model if it has the best F1 score
    joblib.dump(model, 'models/lr/best_logistic_model.pkl')
    joblib.dump(vectorizer, 'models/lr/best_logistic_tokenizer.pkl')
    print(f"Best Logistic Regression model saved with F1 (Validation): {f1_valid:.4f}")

    # Print the F1 score for validation
    print("\nLogistic Regression Model F1 Score (Validation):", f1_valid)

    # Calculate and display training time
    end_time = time.time()  # End the timer
    print(f"Training Time for Logistic Regression: {end_time - start_time:.2f} seconds\n")


def plot_confusion_matrix(cm, model_name):
    """Plot and display the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Pred Negative', 'Pred Positive'],
                yticklabels=['True Negative', 'True Positive'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# RNN Model Training
def rnn_model_training(embedding_dim=128, rnn_units=64, dropout_rate=0.5, batch_size=64, epochs=10, patience=3,
                       learning_rate=0.001):
    """Train and evaluate an RNN model."""
    start_time = time.time()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x_train)  # Use x_train after shuffle
    sequences = tokenizer.texts_to_sequences(x_train)
    max_seq_len = 200
    x_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_len)
    y = y_train  # Use y_train after shuffle

    # Apply tokenizer to validation data
    sequences_valid = tokenizer.texts_to_sequences(x_valid)
    x_padded_valid = tf.keras.preprocessing.sequence.pad_sequences(sequences_valid, maxlen=max_seq_len)

    vocab_size = len(tokenizer.word_index) + 1
    rnn_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len),
        SimpleRNN(rnn_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    rnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = rnn_model.fit(
        x_padded, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_padded_valid, y_valid),  # Using transformed validation data
        callbacks=[early_stopping],
        verbose=1
    )

    # Identify the best epoch based on validation accuracy
    best_epoch = np.argmax(history.history['val_accuracy'])
    print(
        f"Best epoch (based on validation accuracy): {best_epoch + 1} with accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    # Highlight the best epoch based on validation accuracy
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch + 1}')
    plt.annotate(f'Best Epoch: {best_epoch + 1}\nVal Accuracy: {history.history["val_accuracy"][best_epoch]:.4f}',
                 xy=(best_epoch, history.history['val_accuracy'][best_epoch]),
                 xycoords='data', xytext=(0, 40), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='red'), color='red')

    plt.title('RNN Model - Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')

    # Highlight the best epoch based on validation loss
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch + 1}')
    plt.annotate(f'Best Epoch: {best_epoch + 1}\nVal Loss: {history.history["val_loss"][best_epoch]:.4f}',
                 xy=(best_epoch, history.history['val_loss'][best_epoch]),
                 xycoords='data', xytext=(0, -40), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='red'), color='red')

    plt.title('RNN Model - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate model performance
    y_pred_rnn = (rnn_model.predict(x_padded_valid) > 0.5).astype("int32")
    f1_rnn = f1_score(y_valid, y_pred_rnn, average='binary')

    # Calculate confusion matrix
    cm = confusion_matrix(y_valid, y_pred_rnn)
    plot_confusion_matrix(cm, 'RNN')

    # Save the model if it has the best F1 score
    rnn_model.save('models/rnn/best_rnn_model.keras')
    joblib.dump(tokenizer, 'models/rnn/best_rnn_tokenizer.pkl')
    print(f"Best RNN model saved with F1: {f1_rnn:.4f}")

    print("\nRNN Model F1 Score:", f1_rnn)

    # Calculate and display training time
    end_time = time.time()  # End the timer
    print(f"Training Time for RNN: {end_time - start_time:.2f} seconds\n")


# LSTM Model Training
def lstm_model_training(embedding_dim=128, lstm_units=64, dropout_rate=0.5, batch_size=64, epochs=10, patience=3,
                        learning_rate=0.001):
    """Train and evaluate an LSTM model."""
    start_time = time.time()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x_train)  # Use x_train after shuffle
    sequences = tokenizer.texts_to_sequences(x_train)
    max_seq_len = 200
    x_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_len)
    y = y_train  # Use y_train after shuffle

    # Apply tokenizer to validation data
    sequences_valid = tokenizer.texts_to_sequences(x_valid)
    x_padded_valid = tf.keras.preprocessing.sequence.pad_sequences(sequences_valid, maxlen=max_seq_len)

    vocab_size = len(tokenizer.word_index) + 1
    lstm_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = lstm_model.fit(
        x_padded, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_padded_valid, y_valid),  # Using transformed validation data
        callbacks=[early_stopping],
        verbose=1
    )

    # Identify the best epoch based on validation accuracy
    best_epoch = np.argmax(history.history['val_accuracy'])
    print(
        f"Best epoch (based on validation accuracy): {best_epoch + 1} with accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    # Highlight the best epoch based on validation accuracy
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch + 1}')
    plt.annotate(f'Best Epoch: {best_epoch + 1}\nVal Accuracy: {history.history["val_accuracy"][best_epoch]:.4f}',
                 xy=(best_epoch, history.history['val_accuracy'][best_epoch]),
                 xycoords='data', xytext=(0, 40), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='red'), color='red')

    plt.title('LSTM Model - Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')

    # Highlight the best epoch based on validation loss
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch + 1}')
    plt.annotate(f'Best Epoch: {best_epoch + 1}\nVal Loss: {history.history["val_loss"][best_epoch]:.4f}',
                 xy=(best_epoch, history.history['val_loss'][best_epoch]),
                 xycoords='data', xytext=(0, -40), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='red'), color='red')

    plt.title('LSTM Model - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate model performance
    y_pred_lstm = (lstm_model.predict(x_padded_valid) > 0.5).astype("int32")
    f1_lstm = f1_score(y_valid, y_pred_lstm, average='binary')

    # Calculate confusion matrix
    cm = confusion_matrix(y_valid, y_pred_lstm)
    plot_confusion_matrix(cm, 'LSTM')

    # Save the model if it has the best F1 score
    lstm_model.save('models/lstm/best_lstm_model.keras')
    joblib.dump(tokenizer, 'models/lstm/best_lstm_tokenizer.pkl')
    print(f"Best LSTM model saved with F1: {f1_lstm:.4f}")

    print("\nLSTM Model F1 Score:", f1_lstm)

    # Calculate and display training time
    end_time = time.time()  # End the timer
    print(f"Training Time for LSTM: {end_time - start_time:.2f} seconds\n")


def main():
    prepare_data()
    #print("\nTraining Logistic Regression Model...")
    #logistic_regression_model()
    print("\nTraining RNN Model with Fine-Tuning...")
    rnn_model_training(embedding_dim=128, rnn_units=64, dropout_rate=0.5, batch_size=64, epochs=10, patience=3)
    print("\nTraining LSTM Model...")
    lstm_model_training()


if __name__ == "__main__":
    main()
