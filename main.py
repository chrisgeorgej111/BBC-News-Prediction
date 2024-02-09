import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


dataset=pd.read_csv('BBC News Train.csv')

#  Creating new column category_id
dataset['Category_id']=dataset['Category'].factorize()[0]
category = dataset[['Category', 'Category_id']].drop_duplicates().sort_values('Category_id')
# Count of each category
business = dataset[dataset['Category_id'] ==0]
tech=dataset[dataset['Category_id']==1]
politics=dataset[dataset['Category_id']==2]
sports=dataset[dataset['Category_id']==3]
entertainment=dataset[dataset['Category_id']==4]

#  Text Preprocessing
dataset['Text']=dataset['Text'].str.lower()


def text_cleaning(text):
    text = re.sub(r'[^a-zA-Z0-9\']', ' ', text)
    text = text.split()
    text = ' '.join(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))  # Set of English stopwords
    tokens = [i for i in tokens if i not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(i) for i in tokens]

    cleaned_text = " ".join(tokens)
    return cleaned_text


dataset['cleaned_Text'] = dataset['Text'].apply(text_cleaning)

x = dataset['cleaned_Text']
y = dataset['Category_id']
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(dataset['cleaned_Text'])

# Split data into training and testing sets
x = count_matrix  # Features (Bag of Words representation)
y = dataset['Category_id']
  # Target variable
with open('cv.pkl', 'wb') as file:
    pickle.dump(count_vectorizer, file)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=MultinomialNB()
model=model.fit(x_train,y_train)
prediction=model.predict(x_test)
report=classification_report(y_test,prediction)

with open('classifier.pkl','wb') as file:
    pickle.dump(model, file)




text = st.text_area("Enter Text","Type Here")
if text is not None:

    with open('classifier.pkl', 'rb') as file:
        classifier = pickle.load(file)

    with open('cv.pkl', 'rb') as file:
        cv = pickle.load(file)

    input = cv.transform([text]).toarray()

    yy = classifier.predict(input)
    result = ""
    if yy[0]== 0:
        result = "Business News"
    elif yy[0] == 1:
        result = "Tech News"
    elif yy[0] ==2:
        result = "Politics News"
    elif yy[0]== 3:
        result = "Sports News"
    elif yy[0] ==4:
        result = "Entertainment News"

    if st.button("Classify"):
        st.write(result)




