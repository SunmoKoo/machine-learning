# 앙상블 / random forest
from sklearn import datasets
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mnist = datasets.load_digits()
features, labels = mnist.data, mnist.target

def cross_validation(classifier, features, labels):
    cv_score = []
    for i in range(10):
        scores = cross_val_score(classifier, features, labels, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    returen cv_score

dt_cv_scores = corss_validation(tree.DecisionTreeClassifier(), features, labels)

rf_cv_scores = cross_validation(RandomForestClassifier(), features, labels)

cv_list = [['random_forest', rf_cv_scores], ['decision_tree', df_cv_scores],]
df = pd.DataFrame.from_items(cv_list)
df.plot()

np.mean(dt_cv_scores)
np.mean(rf_cv_scores)

