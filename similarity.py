from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Loads required data
def load_data(filename):
    df = pd.read_csv(filename, encoding = 'latin-1')
    return (df['title'],df['subreddit'])

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
        (text,sub) = load_data(i)
        X_list.append(text)
        y_list.append(sub)
    X = pd.concat(X_list)
    y = pd.concat(y_list)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    test_docs = ['Fallout is going to be great!']
    X_test_counts = count_vect.transform(test_docs)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    cosine_similarities = linear_kernel(X_test_tfidf,X_train_tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    print(X.iloc[related_docs_indices[0]])
    print(X.iloc[related_docs_indices[1]])
    print(X.iloc[related_docs_indices[2]])
    print(X.iloc[related_docs_indices[3]])
