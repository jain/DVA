from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.externals import joblib
import pandas as pd
data_dict = {}

def preload_data(subreddit):
    preload_list = []
    data = pd.read_csv("Data/" + subreddit + ".csv", encoding = "latin-1")
    transformed_data = joblib.load("SimData/" + subreddit + "_transformed_data.pkl")
    count_vect = joblib.load("SimData/" + subreddit + "_counts.pkl")
    tfidf_transformer = joblib.load("SimData/" + subreddit + "_tfidf.pkl")
    preload_list.append(data)
    preload_list.append(transformed_data)
    preload_list.append(count_vect)
    preload_list.append(tfidf_transformer)
    data_dict[subreddit] = preload_list

def return_similar_posts(text, subreddit, number_of_similar_posts):
    test_data = []
    test_data.append(text)
    test_counts = data_dict[subreddit][2].transform(test_data)
    test_tfidf = data_dict[subreddit][3].transform(test_counts)
    cosine_similarities = linear_kernel(test_tfidf,data_dict[subreddit][1]).flatten()
    number_of_similar_posts = -1*number_of_similar_posts
    related_docs_indices = cosine_similarities.argsort()[:number_of_similar_posts:-1]
    return [data_dict[subreddit][0].ix[i] for i in related_docs_indices]

if __name__ == '__main__':
    for i in ['books','food','gaming','sports','worldnews',
    'history','askscience','fitness','personalfinance',
    'relationships','technology','Art','movies',
    'Music','space','travel']:
        preload_data(i)
    similar_list = return_similar_posts("Douglas Adams is a great author!", "books", 5)
    print(similar_list)
