import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import svm
import time

# Loads required data
def load_data(filename):
    df = pd.read_csv(filename, encoding = 'latin-1')
    return (df['title'],df['subreddit'])

# Creates a Naive Bayes Classifier
def create_nb_classifier(X,y):
    text_clf = Pipeline([('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf',MultinomialNB()),
        ])
    text_clf = text_clf.fit(X,y)
    return text_clf

# Creates an SVM Classifier
def create_svm_classifier(X,y):
    svm_text_clf = Pipeline([('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
<<<<<<< HEAD
        ('clf', SGDClassifier(loss='log', penalty='l2',alpha=1e-4, n_iter=5, random_state=42)),
=======
        ('clf', SGDClassifier(loss='log', penalty='l2',alpha=1e-4, n_iter=5, random_state=42, n_jobs=-1)),
>>>>>>> 64cf2aa7b6a2c6dc598c493348f2561dbd7c7551
        ])
    svm_text_clf = svm_text_clf.fit(X,y)
    return svm_text_clf

if __name__ == "__main__":
    X_list = []
    y_list = []
    data_file_list = [
    'Data/books.csv',
    'Data/food.csv',
    'Data/gaming.csv',
    'Data/sports.csv',
    'Data/worldnews.csv',
    'Data/history.csv',
    'Data/askscience.csv',
    'Data/fitness.csv',
    'Data/personalfinance.csv',
    'Data/relationships.csv',
    'Data/technology.csv',
    'Data/Art.csv',
    'Data/movies.csv',
    'Data/Music.csv',
    'Data/space.csv',
    'Data/travel.csv'
    ]
    for i in data_file_list:
        print(i)
        (text,sub) = load_data(i)
        X_list.append(text)
        y_list.append(sub)
    X = pd.concat(X_list)
    y = pd.concat(y_list)

    start = time.time()
    text_clf = create_svm_classifier(X,y)
    print (time.time() - start)
    joblib.dump(text_clf,"svm_classifier.pkl")
<<<<<<< HEAD
    scores = cross_validation.cross_val_score(text_clf,X,y,cv = 2)
=======
    print (time.time() - start)
    test_acc = Pipeline([('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', SGDClassifier(loss='log', penalty='l2',alpha=1e-4, n_iter=5, random_state=42, n_jobs=-1)),
        ])
    scores = cross_validation.cross_val_score(test_acc,X,y,cv = 5)
    print (time.time() - start)
>>>>>>> 64cf2aa7b6a2c6dc598c493348f2561dbd7c7551
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

