**SMS Spam Detection Using Machine Learning 📱📊**

**Overview**

This project focuses on building a machine learning model to classify SMS messages as either spam or ham (non-spam). By leveraging various machine learning algorithms, the model aims to accurately predict whether an SMS message is spam based solely on its content.

**Dataset 📂**

We use the SMS Spam Collection Dataset, a popular dataset available on Kaggle. It contains a collection of SMS messages, each categorized as either spam or ham.

Dataset Link: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)

**Features:**

Target Variable: Target label (spam or ham)

Text: Message content

**Problem Statement 💡**

The objective is to:

Preprocess the text data to make it suitable for machine learning.
Train a classification model to differentiate between spam and ham messages.
Evaluate the model's performance on a test dataset.
Build a tool capable of predicting whether a new SMS message is spam or not.

**Workflow 🚀**

1. Data Preprocessing and Exploratory Data Analysis:

Cleaning and tokenizing text.
Removing stopwords and punctuation.
Converting text into numerical features using techniques like TF-IDF.

2. Model Building:

Experimenting with various machine learning algorithms such as Naive Bayes, Logistic Regression, and Support Vector Machines.
Tuning hyperparameters for optimal performance.

3. Evaluation:

Assessing model performance using metrics like accuracy, precision, recall, F1-score, and a confusion matrix.


**Requirements 🛠️**

To run this project, install the following dependencies:

Python 3.7+
NumPy
Pandas
Scikit-learn
NLTK
Matplotlib
WordCloud
