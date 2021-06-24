# -*- coding: utf-8 -*-

#DATA PREPROCESSING

import pandas as pd
import re

# Read data from csv
train = pd.read_csv('train_data.csv', index_col=None)
test = pd.read_csv('test_data.csv')


# Cleaning data by removing unwanted characters
def clean_text(text):
    text_c = re.sub('[.;:!\'?,\"()\[\]*]','',text)
    text_h = re.sub('(<br\s*/><br\s*/>)|(\-)|(\/)',' ',text_c)
    pattern=r'[^a-zA-z0-9\s]'
    text_s=re.sub(pattern,'',text_h)
    text_s = text_s.lower()
    return text_s

train['Reviews'] = train['Reviews'].apply(clean_text)
test['Reviews'] = test['Reviews'].apply(clean_text)


#Remove Stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

train['stopwords_removed'] = train['Reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
test['stopwords_removed'] = test['Reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


#Stemming the text
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

train['stemmed_reviews'] = train['stopwords_removed'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
test['stemmed_reviews'] = test['stopwords_removed'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))


#Tokenizing the text
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

train['lemmatized_reviews'] = train['stopwords_removed'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
test['lemmatized_reviews'] = test['stopwords_removed'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

train_sentiment = train['Sentiment']
test_sentiment = test['Sentiment']

#-----------------------------------------------------------------------------------------------------
# MODEL BUILDING

# Create bag of words from data

#Count Vectoriser using ngrams
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=False, ngram_range=(1,2))
# Stemmed Data
cv_train_s = cv.fit_transform(train['stemmed_reviews'])
cv_test_s = cv.transform(test['stemmed_reviews'])
# Lemmatised data
cv_train_l = cv.fit_transform(train['lemmatized_reviews'])
cv_test_l = cv.transform(test['lemmatized_reviews'])

#Cross Validate Data
cv_X_train_s, cv_X_val_s, cv_y_train_s, cv_y_val_s = train_test_split(cv_train_s, train_sentiment, train_size = 0.8)
cv_X_train_l, cv_X_val_l, cv_y_train_l, cv_y_val_l = train_test_split(cv_train_l, train_sentiment, train_size = 0.8)

#TF-IDF Vectorizer using ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
# Stemmed Data
tv_train_s = tv.fit_transform(train['stemmed_reviews'])
tv_test_s = tv.transform(test['stemmed_reviews'])
# Lemmatised data
tv_train_l = tv.fit_transform(train['lemmatized_reviews'])
tv_test_l = tv.transform(test['lemmatized_reviews'])

#Cross Validate Data
tv_X_train_s, tv_X_val_s, tv_y_train_s, tv_y_val_s = train_test_split(tv_train_s, train_sentiment, train_size = 0.8)
tv_X_train_l, tv_X_val_l, tv_y_train_l, tv_y_val_l = train_test_split(tv_train_l, train_sentiment, train_size = 0.8)


#Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Count Vectorizer
#Stemmed Data using CV
for c in [0.01, 0.05, 0.3, 1]:
    lr_s = LogisticRegression(C=c, max_iter = 200)
    lr_s.fit(cv_X_train_s, cv_y_train_s)
    val_pred = lr_s.predict(cv_X_val_s)
    acc = accuracy_score(cv_y_val_s, val_pred)
    print('Accuracy for lr cv stem ' + str(c) + ' is ' + str(acc))
#Lemmatized data using CV
for c in [0.01, 0.05, 0.3, 1]:
    lr_l = LogisticRegression(C=c, max_iter = 200)
    lr_l.fit(cv_X_train_l, cv_y_train_l)
    val_pred = lr_l.predict(cv_X_val_l)
    acc = accuracy_score(cv_y_val_l, val_pred)
    print('Accuracy for lr cv lem ' + str(c) + ' is ' + str(acc))   

#TFIDF Vectorizer
#Stemmed Data using TFIDF
for c in [0.01, 0.1, 0.3, 1]:
    lr_s = LogisticRegression(C=c, max_iter = 200)
    lr_s.fit(tv_X_train_s, tv_y_train_s)
    val_pred = lr_s.predict(tv_X_val_s)
    acc = accuracy_score(tv_y_val_s, val_pred)
    print('Accuracy for lr tv stem ' + str(c) + ' is ' + str(acc))
#Lemmatized data using TFIDF
for c in [0.01, 0.3, 1]:
    lr_l = LogisticRegression(C=c, max_iter = 200)
    lr_l.fit(tv_X_train_l, tv_y_train_l)
    val_pred = lr_l.predict(tv_X_val_l)
    acc = accuracy_score(tv_y_val_l, val_pred)
    print('Accuracy for lr tv lem ' + str(c) + ' is ' + str(acc)) 
    
    
#SVC classifier model
from sklearn.svm import LinearSVC
#Count Vectorizer
#Stemmed Data using CV
for c in [0.01, 0.1, 0.3, 1]:
    svm_s = LinearSVC(C=c, max_iter = 200)
    svm_s.fit(cv_X_train_s, cv_y_train_s)
    val_pred = svm_s.predict(cv_X_val_s)
    acc = accuracy_score(cv_y_val_s, val_pred)
    print('Accuracy for svc cv stem ' + str(c) + ' is ' + str(acc))
#Lemmatized data using CV
for c in [0.001, 0.05, 0.1, 1]:
    svm_l = LinearSVC(C=c, max_iter = 200)
    svm_l.fit(cv_X_train_l, cv_y_train_l)
    val_pred = svm_l.predict(cv_X_val_l)
    acc = accuracy_score(cv_y_val_l, val_pred)
    print('Accuracy for svc cv lem ' + str(c) + ' is ' + str(acc))   

#TFIDF Vectorizer
#Stemmed Data using TFIDF
for c in [0.01, 0.1, 0.3, 1]:
    svm_s = LinearSVC(C=c, max_iter = 200)
    svm_s.fit(tv_X_train_s, tv_y_train_s)
    val_pred = svm_s.predict(tv_X_val_s)
    acc = accuracy_score(tv_y_val_s, val_pred)
    print('Accuracy for svc tv stem ' + str(c) + ' is ' + str(acc))
#Lemmatized data using TFIDF
for c in [0.01, 0.1, 0.3, 1]:
    svm_l = LinearSVC(C=c, max_iter = 200)
    svm_l.fit(cv_X_train_l, tv_y_train_l)
    val_pred = lr_l.predict(tv_X_val_l)
    acc = accuracy_score(tv_y_val_l, val_pred)
    print('Accuracy for svc tv lem ' + str(c) + ' is ' + str(acc)) 


# Predict the final values
# Logistic Regression using TF vectoriser and lemmatization
final_lr = LogisticRegression(C=1, max_iter = 200)
final_lr.fit(tv_train_l, train_sentiment)
test_pred = final_lr.predict(tv_test_l)
acc = accuracy_score(test_pred, test_sentiment)
print('Accuracy Score for Logistic Regression is ' + str(acc))

# SVC using TF vectoriser and lemmatization
final_svm = LinearSVC(C=0.3, max_iter=200)
final_svm.fit(tv_train_l, train_sentiment)
val_pred = final_svm.predict(tv_test_l)
acc = accuracy_score(val_pred, test_sentiment)
print('Accuracy Score for SVM is ' + str(acc))



