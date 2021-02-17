from django.shortcuts import render
from django.http import JsonResponse
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn import tree
from google_trans_new import google_translator
from sklearn import metrics
import requests
import pyrebase
import tweepy
import pandas as pd
import numpy as np
import emoji
import csv
import re
import nltk
from nltk.corpus import stopwords

firebaseConfig = {
    'apiKey': "AIzaSyCErkdR0G1y05dq5Ea2pavPbC-gTHeyssY",
    'authDomain': "webssru-87cc4.firebaseapp.com",
    'databaseURL': "https://webssru-87cc4.firebaseio.com",
    'projectId': "webssru-87cc4",
    'storageBucket': "webssru-87cc4.appspot.com",
    'messagingSenderId': "231310531528",
    'appId': "1:231310531528:web:4f48608234c255b70d3efd",
    'measurementId': "G-1SNRWYBJD0"
}

firebase = pyrebase.initialize_app(firebaseConfig)

db = firebase.database()

storage = firebase.storage()

def analytic(request):
    messages_th = ""
    messages_en = ""
    acc = []
    result = []
    result_ssense = []
    score = []
    listP = []
    listSs = []
    requirement = db.child("Requirement").get()
    for require in requirement.each():
        message = require.val()
    messages_th = message["comments_th"]
    messages_en = message["comments_en"]
    textBlob_clf(messages_en, acc, result)
    ssense(messages_th, score, result_ssense, listSs)
    showAnalysis(listP)
    pos = listP[0]
    neu = listP[1]
    neg = listP[2]
    acc_sc = acc[0]
    sc = score[0]
    result_sc = result[0][0]
    result_ss = result_ssense[0]
    preneg = listSs[0]
    prepos = listSs[1]
    preseg = listSs[2]
    prekey = listSs[3]
    return render(request, 'analytic.html', {"pos": pos, "neu": neu, "neg": neg, "acc": acc_sc, "result_sc": result_sc,
                                            "messages": messages_th, "score": sc, 
                                            "result_ss":result_ss, "preneg":preneg, 
                                            "prepos":prepos, "preseg":preseg, "prekey":prekey})

def textBlob_clf(messages_en, acc, result):
    # download Data
    download_csv()

    try:
        # Set Data
        df_Clean = pd.read_csv('twitterCrawler_clean.csv')
        df_Clean['Text'] = df_Clean['text']
        df_Clean['Polarity'] = df_Clean['Text'].apply(getPolarity)
        df_Clean['Analysis'] = df_Clean['Polarity'].apply(getAnalysis)
        dataSet = df_Clean[df_Clean.Polarity != 0]
        dataSet['Analysis'] = dataSet['Analysis'].replace('Positive', 1)
        dataSet['Analysis'] = dataSet['Analysis'].replace('Negative', 0)
        dataSet = dataSet.drop(['Unnamed: 0', 'text', 'Polarity'], axis=1)
        df_Sentiment = pd.read_csv('twitterCrawler_Sentiment_final.csv')
        df_Sentiment = df_Sentiment.drop(['Unnamed: 0'], axis=1)
        df_Sentiment = df_Sentiment.append(dataSet, ignore_index = True)

        # Analysis_Data
        stopset = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
        y = df_Sentiment["Analysis"]
        X = vectorizer.fit_transform(df_Sentiment["Text"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        svm_clf = SVC(probability=True, kernel='linear')
        svm_clf.fit(X_train, y_train)
        pred = svm_clf.predict(X_test)
        message_input = np.array([messages_en])
        message_vector = vectorizer.transform(message_input)
        result.append(svm_clf.predict(message_vector))
        acc_sc = "%.2f"%(metrics.accuracy_score(y_test, pred))
        acc.append(acc_sc)
    except MemoryError:
        # Set Data
        df_Sentiment = pd.read_csv('twitterCrawler_Sentiment_final.csv')
        df_Sentiment = df_Sentiment.drop(['Unnamed: 0'], axis=1)

        # Analysis_Data
        stopset = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
        y = df_Sentiment["Analysis"]
        X = vectorizer.fit_transform(df_Sentiment["Text"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        svm_clf = SVC(probability=True, kernel='linear')
        svm_clf.fit(X_train, y_train)
        pred = svm_clf.predict(X_test)
        message_input = np.array([messages_en])
        message_vector = vectorizer.transform(message_input)
        result.append(svm_clf.predict(message_vector))
        acc_sc = "%.2f"%(metrics.accuracy_score(y_test, pred))
        acc.append(acc_sc)

def ssense(messages, score, result_ssense, listSs):
    url = "https://api.aiforthai.in.th/ssense"

    text = messages

    params = {'text': text}

    headers = {
        'Apikey': "ARVYukGnRlOej6pT7BIxKd993BVxaf37"
    }

    response = requests.get(url, headers=headers, params=params)

    response_dict = response.json()
    scores = response_dict['sentiment']['score']
    polarity = response_dict['sentiment']['polarity']
    listSs.append(response_dict['preprocess']['neg'])
    listSs.append(response_dict['preprocess']['pos'])
    listSs.append(response_dict['preprocess']['segmented'])
    listSs.append(response_dict['preprocess']['keyword'])
    score.append(scores)
    result_ssense.append(polarity)

def showAnalysis(listP):
    # Set Data
    df_Clean = pd.read_csv('twitterCrawler_clean.csv')
    df_Clean['Text'] = df_Clean['text']
    df_Clean['Polarity'] = df_Clean['Text'].apply(getPolarity)
    df_Clean['Analysis'] = df_Clean['Polarity'].apply(getAnalysis)
    dataSet = df_Clean[df_Clean.Polarity != 0]
    dataSet['Analysis'] = dataSet['Analysis'].replace('Positive', 1)
    dataSet['Analysis'] = dataSet['Analysis'].replace('Negative', 0)
    dataSet = dataSet.drop(['Unnamed: 0', 'text', 'Polarity'], axis=1)
    anlysis_df = pd.read_csv('twitterCrawler_Sentiment_final.csv')
    anlysis_df = anlysis_df.drop(['Unnamed: 0'], axis=1)
    anlysis_df = anlysis_df.append(dataSet, ignore_index = True)
    anlysis_df['Polarity'] = anlysis_df['Text'].apply(getPolarity)
    anlysis_df['Analysis'] = anlysis_df['Polarity'].apply(getAnalysis)
    # Get the percentage of positive tweets
    postweets = anlysis_df[anlysis_df.Analysis == 'Positive']
    postweets = postweets['Text']
    posPercen = round((postweets.shape[0] / anlysis_df.shape[0]) * 100, 1)
    listP.append(posPercen)

    # Get the percentage of neutral tweets
    neutweets = anlysis_df[anlysis_df.Analysis == 'Neutral']
    neutweets = neutweets['Text']
    neuPercen = round((neutweets.shape[0] / anlysis_df.shape[0]) * 100, 1)
    listP.append(neuPercen)

    # Get the percentage of negative tweets
    negtweets = anlysis_df[anlysis_df.Analysis == 'Negative']
    negtweets = negtweets['Text']
    negPercen = round((negtweets.shape[0] / anlysis_df.shape[0]) * 100, 1)
    listP.append(negPercen)
    # return render(request, 'analytic.html', {"listP": listP})

def download_csv():
    filename_new = "twitterCrawler_Sentiment_final.csv"
    cloud = "twitterCrawler_Sentiment_final.csv"
    storage.child(cloud).download("", filename_new)

# Create a function th get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create a function to computer the negative, neutral and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'