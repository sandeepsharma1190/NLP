# NLP - Natural Language Processing

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

line1 = 'We are learning Python. This is second line'

sentence = nltk.sent_tokenize(line1)
word = nltk.word_tokenize(line1)

# List of stop words
stopwords.words('english')
stopwords.words('spanish')

##############################################
# Stemmer
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    words = [stemmer.stem(word) for word in words if word not in set (stopwords.words('english'))]
    sentence[i] = ' '.join(words)

##############################################
# Lemmatizer
lemmatizer = WordNetLemmatizer()
for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentence[i] = ' '.join(words)  
##############################################

# Cleaning the texts
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
corpus = []
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

#######################=======================================================================
# TF-IDF = Term Frequency - Inverse document frequency
# Using Lemmatizer with TF-IDF Model
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review) 
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()









































