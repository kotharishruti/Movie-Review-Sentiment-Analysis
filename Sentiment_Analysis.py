#!/usr/bin/env python
# coding: utf-8

#Importing libraries
import os
import re
import logging
import pandas as pd
import numpy as np
import nltk.data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
from sklearn import naive_bayes, svm, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import scikitplot as skplt
import matplotlib.pyplot as plt
import time

#Pre-processing movie reviews
def clean_review(raw_review):
    # Remove HTML markup
    text = BeautifulSoup(raw_review,features="html.parser")
    
    #Removing digits and punctuation
    text = re.sub("[^a-zA-Z]", " ", text.get_text())
    
    #Converting to lowercase
    text = text.lower().split()
    
    # Removing stopwords
    stops = set(stopwords.words("english"))
    words = [w for w in text if w not in stops]
    
    # Return a cleaned string
    return " ".join(words)


#Generates a feature vector(word2vec averaging) for each movie review
def review_to_vec(words, model, num_features):
    """
    This function generates a feature vector for the given review.
    Input:
        words: a list of words extracted from a review
        model: trained word2vec model
        num_features: dimension of word2vec vectors
    Output:
        a numpy array representing the review
    """
    
    feature_vec = np.zeros((num_features), dtype="float32")
    word_count = 0
    
    # index2word_set is a set consisting of all words in the vocabulary
    index2word_set = set(model.index2word)
    
    for word in words:
        if word in index2word_set: 
            word_count += 1
            feature_vec += model[word]

    feature_vec /= word_count
    return feature_vec
    

#Generates vectorized movie reviews
def gen_review_vecs(reviews, model, num_features):
    """
    Function which generates a m-by-n numpy array from all reviews,
    where m is len(reviews), and n is num_feature
    Input:
            reviews: a list of lists. 
                     Inner lists are words from each review.
                     Outer lists consist of all reviews
            model: trained word2vec model
            num_feature: dimension of word2vec vectors
    Output: m-by-n numpy array, where m is len(review) and n is num_feature
    """

    curr_index = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:

       if curr_index%1000 == 0.:
           print ("Vectorizing review %d of %d" % (curr_index, len(reviews)))
   
       review_feature_vecs[curr_index] = review_to_vec(review, model, num_features)
       curr_index += 1
       
    return review_feature_vecs

#TFIDF vectorization
def tfidf_vectorizer(train_list,test_list,train_data,test_data):
    for i in range(0, len(train_data.review)):
        
        # Append raw texts as TFIDF vectorizers take raw texts as inputs
        train_list.append(clean_review(train_data.review[i]))
        if i%1000 == 0:
            print ("Cleaning training review", i)

    for i in range(0, len(test_data.review)):
        
        # Append raw texts as TFIDF vectorizers take raw texts as inputs
        test_list.append(clean_review(test_data.review[i]))
        if i%1000 == 0:
            print ("Cleaning test review", i)
    count_vec = TfidfVectorizer(analyzer="word", max_features=10000, ngram_range=(1,2), sublinear_tf=True)
    print ("Vectorizing input texts")
    train_vec = count_vec.fit_transform(train_list)
    test_vec = count_vec.transform(test_list)
    return train_vec,test_vec,count_vec

#Performing dimensionality reduction using SelectKBest
def dimensionality_reduction(train_vec,test_vec,y_train_data):
    print ("Performing feature selection based on chi2 independence test")
    fselect = SelectKBest(chi2,k=500)
    train_vec = fselect.fit_transform(train_vec, y_train_data)
    test_vec = fselect.transform(test_vec)
    return train_vec,test_vec

#Multinomial Naive Bayes classifier
def naive_bayes(train_vec,test_vec,y_train_data):
    start = time.time()
    nb = MultinomialNB()
    cv_score = cross_val_score(nb, train_vec,y_train_data, cv=10)
    print("Training Multinomial Naive Bayes")
    nb = nb.fit(train_vec,y_train_data)
    pred_naive_bayes = nb.predict(test_vec)
    print ("CV Score = ", cv_score.mean())
    print ("Total time taken for Multinomial Naive Bayes is ", time.time()-start, " seconds")
    return pred_naive_bayes

#Random Forest classifier
def random_forest(train_vec,test_vec,y_train_data):
    start = time.time()
    rfc = RFC(n_estimators = 100,oob_score = True,max_features ="auto")
    print("Training %s" % ("Random Forest"))
    rfc = rfc.fit(train_vec,y_train_data)
    print("OOB Score =", rfc.oob_score_)
    pred_random_forest = rfc.predict(test_vec)
    print ("Total time taken for Random Forest is ", time.time()-start, " seconds")
    return pred_random_forest

#Linear SVC classifier
def linear_svc(train_vec,test_vec,y_train_data): 
    start = time.time()
    svc = svm.LinearSVC()
    param = {'max_iter':[1000,2000],'C': [1e15,1e13,1e11,1e9,1e7,1e5,1e3,1e1,1e-1,1e-3,1e-5]}
    print ("Training SVC")
    svc = GridSearchCV(svc, param,cv=10)
    svc = svc.fit(train_vec, y_train_data)
    pred_linear_svc = svc.predict(test_vec)
    print ("Optimized parameters:", svc.best_estimator_)
    print ("Best CV score:", svc.best_score_)
    print ("Total time taken for Linear SVC is ", time.time()-start, " seconds")
    print("Generating confusion matrix")
	#Below confusion matrix code is commented as it takes a lot of time to run. The plots have been added in the project report.
    #predictions = cross_val_predict(svc, train_vec, y_train_data)
    #skplt.metrics.plot_confusion_matrix(y_train_data, predictions)
    #plt.show()
    return pred_linear_svc

#Logistic Regression
def logistic_regression(train_vec,test_vec,y_train_data):
    start = time.time()
    clf = LogisticRegression(random_state=0, solver='lbfgs',max_iter=10000,multi_class='multinomial')
    cv_score = cross_val_score(clf, train_vec,y_train_data, cv=10)
    print("Training Logistic Regression")
    clf = clf.fit(train_vec,y_train_data)
    pred_logistic= clf.predict(test_vec)
    print ("CV Score = ", cv_score.mean())
    print ("Total time taken for Logistic is ", time.time()-start, " seconds")
    print("Plotting Precision recall curve")
    result = clf.predict_proba(train_vec)
    skplt.metrics.plot_precision_recall(y_train_data,result)
    plt.show()
    return pred_logistic

#Word2Vec vectorization
def word2vec(train_data,test_data,train_list,test_list):
    model_name = "GoogleNews-vectors-negative300.bin.gz"
    model_type = "bin"
    num_features = 300
    for i in range(0, len(train_data.review)):
        train_list.append(clean_review(train_data.review[i]))
        if i%1000 == 0:
            print("Cleaning training review",i)
    for i in range(0, len(test_data.review)):
        test_list.append(clean_review(test_data.review[i]))
        if i%1000 == 0:
            print ("Cleaning test review", i)
    print ("Loading the pre-trained model")
    #The below part has been commented as the model was loaded, movie reviews were vectorized and stored in below pkl files, 
    #as this takes a lot of time to execute. 
    #We are reading the pkl files to get the final vectorized data
    
    #model = Word2Vec.load_word2vec_format(model_name, binary=True)
    print ("Vectorizing training review")
    #train_vec = gen_review_vecs(train_list, model, num_features)
    #print ("Vectorizing test review")
    #test_vec = gen_review_vecs(test_list, model, num_features)
    
    #print("Writing to DataFrame after vectorizing")
    #df_train = pd.DataFrame(train_vec)
    #df_test = pd.DataFrame(test_vec)
    #df_train.to_pickle("train.pkl")
    #df_test.to_pickle("test.pkl")
    
    
    y_train_data = train_data.sentiment
    train_df= pd.read_pickle("train.pkl")
    test_df = pd.read_pickle("test.pkl")
    
    #Word2Vec cannot be used with Multinomial Naive Bayes as Multinomial Naive Bayes does not work with negative values 
    pred_logistic = logistic_regression(train_df,test_df,y_train_data)
    pred_random_forest = random_forest(train_df,test_df,y_train_data)
    pred_linear_svc = linear_svc(train_df,test_df,y_train_data)
    
    output = pd.DataFrame(data = {"id": test_data.id,"review":test_data.review, "sentiment": pred_linear_svc})
    output.to_csv("word2vec_svc.csv", index=False)
	
#Testing a custom movie review
def test_custom_review(count_vec,train_vec,y_train_data):
    print('\nTest a custom review message')
    print('Enter review to be analysed: ',end=" ")

    test = []
    test_list = []
    test.append(input())
    test_review= pd.DataFrame(data = {"id": 1, "review": test})
    print("Cleaning the test review")
    for i in range(0, len(test_review.review)):
        test_list.append(clean_review(test_review.review[i]))
    print("Vectorizing the test review")
    test_review_vec = count_vec.transform(test_list)
    print("Predicting")
    pred_naive_bayes= naive_bayes(train_vec,test_review_vec,y_train_data)
    if(pred_naive_bayes == 1):
        print("The review is predicted positive")
    else:
        print("The review is predicted negative")
    
    
if __name__ == "__main__":
    train_list = []
    test_list = []
    word2vec_input = []

    pred_naive_bayes = []
    pred_logistic = []
    pred_random_forest = []
    pred_linear_svc = []
    train_data = pd.read_csv("labeledTrainData.tsv",header=0, delimiter="\t", quoting=0)
    test_data = pd.read_csv("testData.tsv",header=0, delimiter="\t", quoting=0)

    y_train_data = train_data.sentiment

    #Vectorization - TFIDF
    print("Using TFIDF ")
    train_vect,test_vec,count_vec= tfidf_vectorizer(train_list,test_list,train_data,test_data)

    #Dimensionality Reduction
    train_vec,test_vec = dimensionality_reduction(train_vect,test_vec,y_train_data)
    
    #Prediction 
    pred_naive_bayes = naive_bayes(train_vec,test_vec,y_train_data)
    pred_random_forest = random_forest(train_vec,test_vec,y_train_data)
    pred_linear_svc = linear_svc(train_vec,test_vec,y_train_data)
    pred_logistic = logistic_regression(train_vec,test_vec,y_train_data)      

    #Writing output of classifier with highest accuracy(Linear SVC)to csv 
    output = pd.DataFrame(data = {"id": test_data.id,"review":test_data.review, "sentiment": pred_linear_svc})
    output.to_csv("tfidf_svc.csv", index=False)

    print("Using pre-trained word2vec model")
    train_list = []
    test_list = []
    pred_logistic = []
    pred_random_forest = []
    pred_linear_svc = []

    word2vec(train_data,test_data,train_list,test_list)

    #Test a custom review using Multinomial Naive Bayes
    test_custom_review(count_vec,train_vect,y_train_data)
