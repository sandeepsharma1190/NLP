# NLP

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

# Importing the dataset
dataset = pd.read_excel('file')

def missing_values(df):    
    print("Number of records with missing location:",dataset['Admin Comments'].isnull().sum())
    print("Number of records with missing keywords:",dataset['Reason'].isnull().sum())
    print('{}% of location values are missing from Total Number of Records.'.format(round((dataset['Admin Comments'].isnull().sum())/(df.shape[0])*100),2))
    print('{}% of keywords values are missing from Total Number of Records.'.format(round((dataset['Reason'].isnull().sum())/(df.shape[0])*100),2))
missing_values(dataset)

# dataset.isna().sum()

dataset1 = dataset.dropna(how = 'all')

dataset1 = dataset.fillna({'Admin Comments': 'Comment not Available',
                           'Reason':'N/A'})
# dataset1.isna().sum()

Error_code = {'N/A': 0,
              'No Issue Found':0,
              'Attachment Missing': 1,
              'Avaya ID provided is incorrect': 1,
              'Business justification required': 1,
              'Correct Template Not Used': 1,
              'Data warehouse Call - Missing details' : 1,
              'Incorrect Business Site Request assign to Qfiniti': 1,
              'Request submitted wrong intake form':1,
              'SITE Name Missing':1,
              'Unable to locate profile':1}

dataset1['Reason'] = [Error_code[item] for item in dataset1['Reason']] 

dataset.info()

dataset_char_count = dataset1['Admin Comments'].apply(lambda x: len(str(x)))
dataset_unique_word_count = dataset1['Admin Comments'].apply(lambda x: len(set(str(x).split())))
dataset_punctuation_count = dataset1['Admin Comments'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

unique_words = dataset['Admin Comments'].unique()
dataset['Admin Comments'].nunique()

# List of stop words
stopwords.words('english')

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 6507):
    review = re.sub('[^a-zA-Z]', ' ', dataset1['Admin Comments'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Creating the TF-IDF model
# from sklearn.feature_extraction.text import TfidfVectorizer
# cv1 = TfidfVectorizer()
# X1 = cv1.fit_transform(corpus).toarray()
# y = dataset1.iloc[:, 1].values

# Creating the Bag of Words model
print (len(X[0]))
# Sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3530)
X = cv.fit_transform(corpus).toarray()
y = dataset1.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#########======================
print('Training Set Shape = {}'.format(X_train.shape))
print('Test Set Shape = {}'.format(X_test.shape))
#########======================

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, rfc_pred)
print(cm)
print (accuracy_score(y_test, rfc_pred))

###########========================
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rfc, X = X_test, y = y_test, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rfc, X = X, y = y, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))




















