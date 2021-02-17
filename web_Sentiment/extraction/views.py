from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
import pandas as pd
import numpy as np
import pyrebase
import emoji
import csv
import re

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

storage = firebase.storage()

data = []

def cleanTxt(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                  '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)  # Removing @mentions
    text = re.sub('RT', '', text)
    text = re.sub('(@[A-Za-z0-9_]+)', '', text)
    text = re.sub('(#[A-Za-z0-9_]+)', '', text)
    text = re.sub('“', '', text)
    text = re.sub('”', '', text)
    text = re.sub('’', '', text)
    text = re.sub('—', '', text)
    text = emoji_pattern.sub(r'', text)
    text = re.sub('@[A-Za-z0–9]+', '', text)
    return text


def extraction(request):
    filename = "twitterCrawler.csv"
    df = pd.read_csv(filename)
    df['text'] = df['text'].apply(cleanTxt)
    filename_clean = df["text"].to_csv('twitterCrawler_clean.csv')
    cloud = "twitterCrawler_clean.csv"
    storage.child(cloud).put(filename_clean)
    data = df['text']
    return render(request, 'extraction.html', {"data": data})
