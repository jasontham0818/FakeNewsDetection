# Fake News Detection Project

This repository contains the implementation of a Fake News Detection system, including data preprocessing, model training, testing, and evaluation. The project leverages Python, TensorFlow, and other popular libraries for building machine learning models.

---

## Project Structure

The project consists of the following files:

- **`DataPreprocessing.py`**: This script handles all data preprocessing tasks, such as cleaning the data, tokenizing text, and splitting the dataset into training, validation, and testing sets.
  
- **`main.py`**: This script is responsible for training the machine learning models (Logistic Regression, RNN, LSTM) on the preprocessed data. It also saves the trained models for later use.
  
- **`model_testing.py`**: This script tests the performance of the trained models on the test dataset and evaluates them using various metrics such as accuracy, precision, recall, and F1 score.

- **`interface.py`**: This script contains the implementation of the user-friendly interface for interacting with the fake news detection system. It provides a way to input new articles and obtain predictions from the trained models.

- **`requirements.txt`**: This file lists all the dependencies required to run the project. You can install them using `pip install -r requirements.txt`.

- **`data directory`**: Consists of all of the dataset used in this project.

- **`models directory`**: All of the models trained for the Fake News Detection.

- **`result directory`**: Result of the accuracy of each models.

## Getting Started

### Prerequisites

- Python 3.12
- Jupyter Notebook
- Required libraries (install using the command below):
  ```bash
    pip install -r requirements.txt
  
### Data Preprocessing

- Run the `DataPreprocessing.py` script to preprocess the data:

  ```bash
    python DataPreprocessing.py 

### Training Models

- Run the 'main.py' script to train the models:

  ```bash
    python main.py
  
### Testing of the models

- Run the 'model_testing.py' script to test the models:
  ```bash
    python model_testing.py
  
### Interface development

- Run the 'interface.py' script for the user-friendly interface:
  ```bash
    python interface.py
  
### Overview of the whole project

- For a detailed overview and implementation, please refer to the Jupyter Notebook file located in the repository.
  

