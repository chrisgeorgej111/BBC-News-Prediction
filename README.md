# BBC-News-Prediction
News Prediction

Building a news classification system involves several steps, including  data
preprocessing, and model training. Below is a high-level outline of the steps you can follow
using Natural Language Processing (NLP) techniques:

**1. Data Cleaning and Preprocessing:**
- Removed any irrelevant information, such as HTML tags, advertisements, or non-text content.
- Tokenized the text (split it into words or subwords) and remove stop words.
- Performed lemmatization or stemming to reduce words to their base form.
- Handled missing data and ensure a consistent format.
**2. Text Representation:**
- Converted the text data into numerical format suitable for machine learning models using count vectorizer.
**3. Topic Labeling:**
- Manually inspected a sample of articles in each cluster to assign topic labels. This step helps in
labeling the clusters with meaningful topics.
**4. Classification Model:**
- Split the data into training and testing sets.
- Trained a supervised machine learning model (e.g., Naive Bayes, Support Vector Machines, or
deep learning models like LSTM or BERT) to predict the topic of a news article.

Naviv Bayes model gave an accuracy of 98%.. This has been used to develop an streamlit app to predict the type of news.
