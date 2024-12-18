import streamlit as st
import requests
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns 
import nltk
import regex as re
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report ,confusion_matrix,precision_score

nltk.download('punkt_tab')

def Text_preprocessing(text):
    text =  text.lower()  
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]  

    stemmer = PorterStemmer()
    
    text = [stemmer.stem(word)for word in text]   
    
    return " ".join(text)
    

tfidf = pickle.load(open("Vectorizers.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))


st.title("Email Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    ## Steps 
    # 1 . Preprocess

    Transformed_sms=Text_preprocessing(input_sms)

    # 2 . Vectorizer

    vector_input = tfidf.transform([Transformed_sms])
    # 3 . Predict

    prediction = model.predict(vector_input)[0]
    # 4 . Display

    if prediction == 1: 
        st.header("Spam")
    else:
        st.header("Not Spam")



