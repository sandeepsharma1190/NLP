# Spam Collection 

# https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

# from nltk.stem import PorterStemmer

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('SMSSpamCollection', sep = '\t', names = ['label','message'])

# Data Cleaning
# stopwords.words('english')// set(stopwords.words('english'))// set(all_stopwords)
corpus = []
ps = PorterStemmer()
for i in range (0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['message'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the TF-IDF model
# from sklearn.feature_extraction.text import TfidfVectorizer
# cv1 = TfidfVectorizer()
# X1 = cv1.fit_transform(corpus).toarray()
# y = dataset1.iloc[:, 1].values

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(dataset['label'])
y=y.iloc[:,1].values

# Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#########======================
print('Training Set Shape = {}'.format(X_train.shape))
print('Test Set Shape = {}'.format(X_test.shape))
#########======================

# Support Vector with Linear Kernel (Best Accuracy)=========================================================
from sklearn.svm import SVC
svc = SVC(kernel = 'linear').fit(X_train, y_train)
pred = svc.predict(X_test)
print (confusion_matrix(y_test, pred))
print (accuracy_score(y_test, pred))
# 0.9883408071748879

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = nbsvc X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svc, X = X_test, y = y_test, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svc, X = X, y = y, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Multinomial NB =================================================================
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
pred = nb.predict(X_test)
print (confusion_matrix(y_test, pred))
print (accuracy_score(y_test, pred))
# 0.9811659192825112
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = nb, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = nb, X = X_test, y = y_test, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = nb, X = X, y = y, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Boosting=====================================================================================================
from xgboost import XGBClassifier
xgb = XGBClassifier().fit(X_train, y_train)
pred = xgb.predict(X_test)
print (confusion_matrix(y_test, pred))
print (accuracy_score(y_test, pred))
# 0.9856502242152466

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb, X = X_test, y = y_test, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb, X = X, y = y, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))














