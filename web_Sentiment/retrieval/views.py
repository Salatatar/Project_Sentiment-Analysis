from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
import matplotlib.pyplot as plt
import pyrebase
import pandas as pd
import csv

data = []

# Create your views here
def retrieval(request):
    filename = "twitterCrawler.csv"
    df = pd.read_csv(filename)
    data = df["text"]
    return render(request, 'retrieval.html', {"data": data})