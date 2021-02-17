from django.shortcuts import render
from django.http import JsonResponse

def knn(request):
    return render(request, 'knn.html', {})

def naive_Bayes(request):
    return render(request, 'naive_Bayes.html', {})

def decision_Tree(request):
    return render(request, 'decision_Tree.html', {})

def random_Forest(request):
    return render(request, 'random_Forest.html', {})

def svm(request):
    return render(request, 'svm.html', {})