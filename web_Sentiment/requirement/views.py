from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
from google_trans_new import google_translator
# from googletrans import Translator
# import goslate
import pyrebase
import tweepy
import pandas as pd
import numpy as np
import emoji
import csv
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud
import json

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

detector = google_translator()
translator = google_translator()

def requirement(request):
    return render(request, 'requirement.html', {})


def submitRequire(request):
    sport = request.GET['sports']
    comment = request.GET['comments']
    if {'sports': sport} != "" and {'comments': comment} != "":
        detect_result = detector.detect(comment)
        source = comment
        if detect_result[0] == "th":
            try:
                translate_text = translator.translate(source, lang_tgt='en') 
            except:
                print ("กรุณาลองใหม่ !!!")
            data = {"sports": {'sports': sport},
                    "comments_th": comment, 
                    "comments_en": translate_text}
            db.child("Requirement").push(data)
        else:
            data = {"sports": sport, "comments_th": comment}
            db.child("Requirement").push(data)
        search = data["sports"]["sports"]
        save_data(search)
        return render(request, 'requirement.html', {'sports': sport, 'comments': comment, 'alert_flag': True})
    return render(request, 'requirement.html', {})


def save_data(search):
    # Create the authentication object
    authenticate = tweepy.OAuthHandler(
        "r5kQ6RXk4r3aYCCAbUsOt6PWT", "FndtM1xO2ZFMiI3W8WTLdu97AyuoRc4SiPv4ZSMC0uEVduDK3g")
    # Set the access token and access token secret
    authenticate.set_access_token(
        "1232992320934465537-0sT0CbXCLLEqWoClkiiNs9XUR1yd1W", "EXuvRYISh8tWQoDdcsH0EbGS9ttH9qRYjKAxUIMpOi37O")

    # Create the API object while passing in the auth information
    api = tweepy.API(authenticate, wait_on_rate_limit=True)
    query = "#" + search

    df = pd.DataFrame(columns=["create_at", "text",
                               "hashtag", "retweet_count"])
    # for tweet in tweepy.Cursor(api.search, q=query, count=1000, result_type="recent", tweet_mode='extended').items():
    for tweet in tweepy.Cursor(api.search, q=query, count=1000, lang="en", result_type="recent", tweet_mode='extended').items():
        entity_hashtag = tweet.entities.get('hashtags')
        hashtag = ""
        for i in range(0, len(entity_hashtag)):
            hashtag = hashtag + "/" + entity_hashtag[i]["text"]
        re_count = tweet.retweet_count
        create_at = tweet.created_at
        try:
            text = tweet.retweeted_status.full_text
        except:
            text = tweet.full_text
        new_column = pd.Series(
            [create_at, text, hashtag, re_count], index=df.columns)
        df = df.append(new_column, ignore_index=True)
    filename = df["text"].to_csv('twitterCrawler.csv')
    cloud = "twitterCrawler.csv"
    storage.child(cloud).put(filename)
