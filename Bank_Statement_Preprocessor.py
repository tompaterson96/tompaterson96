import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def preprocess(file_path = os.getcwd() + "\\" + "BankStatementDataset.csv", max_df = 0.5):

    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project

    file_path = os.getcwd() + "\\" + "BankStatementDataset.csv"
    #print("Accessing file at: " + file_path)

    dataset = pd.read_csv(file_path)
    array = dataset.values
    #values = array[:,0]
    references = array[:,1]
    #features = array[:,0:2]
    labels = array[:,2]

    ### PROCEEDING WITH ONLY REFERENCES AS FEATURES

    features_train, features_test, labels_train, labels_test = train_test_split(references, labels, test_size=0.1, random_state=1)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=max_df,
                                    stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)

    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    #selector = SelectPercentile(f_classif, percentile=10)
    #selector.fit(features_train_transformed, labels_train)
    #features_train_transformed = selector.transform(features_train_transformed)
    #features_test_transformed  = selector.transform(features_test_transformed)

    ### info on the data
    #print "no. of Chris training emails:", sum(labels_train)
    #print "no. of Sara training emails:", len(labels_train)-sum(labels_train)

    return features_train_transformed, features_test_transformed, labels_train, labels_test

