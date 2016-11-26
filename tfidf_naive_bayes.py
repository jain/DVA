# SELECT * FROM (SELECT ROW_NUMBER() OVER(), created_utc,subreddit,title
# FROM [fh-bigquery:reddit_posts.2016_08]
# WHERE subreddit = 'gaming') WHERE f0_ >= 20001 AND f0_ <= 30000;

# DIY.csv, AskReddit.csv, gadgets.csv, movies.csv, news.csv, space.csv,
# television.csv, todayilearned.csv

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import svm

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
        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-4, n_iter=5, random_state=42)),
        ])
    svm_text_clf = svm_text_clf.fit(X,y)
    return svm_text_clf

if __name__ == "__main__":
    X_list = []
    y_list = []
    data_file_list = [
    'Data/Subreddits/books.csv',
    'Data/Subreddits/food.csv',
    'Data/Subreddits/gaming.csv',
    'Data/Subreddits/sports.csv',
    'Data/Subreddits/worldnews.csv',
    'Data/Subreddits/art.csv',
    'Data/Subreddits/music.csv',
    'Data/Subreddits/history.csv',
    'Data/Subreddits/askscience.csv',
    'Data/Subreddits/Fitness.csv',
    'Data/Subreddits/personalfinance.csv',
    'Data/Subreddits/relationships.csv',
    'Data/Subreddits/technology.csv',
    ]
    for i in data_file_list:
        print(i)
        (text,sub) = load_data(i)
        X_list.append(text)
        y_list.append(sub)
    X = pd.concat(X_list)
    y = pd.concat(y_list)
    text_clf = create_svm_classifier(X,y)
    joblib.dump(text_clf,"svm_classifier.pkl")
    scores = cross_validation.cross_val_score(text_clf,X,y,cv = 5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

