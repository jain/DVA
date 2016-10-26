import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_data(filename):
    df = pd.read_csv(filename)
    return (df['title'],df['subreddit'])

def create_tfidf(X):
    tf = TfidfVectorizer()
    X_tfidf = tf.fit_transform(X)
    return (tf,X_tfidf)

def create_model(X,y):
    clf = MultinomialNB().fit(X,y)
    return clf

def find_subreddit(tf,clf,X_test):
    td = tf.transform(X_test)
    return clf.predict(td)

if __name__ == "__main__":
    X_list = []
    y_list = []
    for i in ['gatech-aug2016.csv','joker-aug2016.csv']:
        (text,sub) = load_data(i)
        X_list.append(text)
        y_list.append(sub)
    X = pd.concat(X_list)
    y = pd.concat(y_list)
    (tf,X) = create_tfidf(X)
    clf = create_model(X,y)
    results = find_subreddit(tf,clf,['Suicide Squad','UGA is an absolute joke','Tech is amazing!','Heath Ledger','The Killing Joke','College is expensive'])
    print(results)
