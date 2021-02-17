from django.urls import path
from comparison import views

urlpatterns = [
    path('KNN', views.knn, name='knn'),
    path('Naive-Bayes', views.naive_Bayes, name='naiveBayes'),
    path('Decistion-Tree', views.decision_Tree, name='decistionTree'),
    path('Random-Forest', views.random_Forest, name='randomForest'),
    path('SVM', views.svm, name='svm'),
]