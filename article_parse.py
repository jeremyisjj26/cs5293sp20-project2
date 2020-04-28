import glob
import json
import csv
import pandas as pd
import nltk
import re
import numpy as np

outputFile = open("convertedData.csv","w")

outputWriter = csv.writer(outputFile)

outputWriter.writerow(["text","cite_spans","ref_spans","section"])

data_folder = "test_json"

for filename in glob.iglob(data_folder+"/*.json"):

        sourceFile = open(filename, "r")

        json_data = json.load(sourceFile)

        for body_text in json_data["body_text"]:
                newRow = []
                for attribute in body_text:
                        newRow.append(body_text[attribute])
                outputWriter.writerow(newRow)


sourceFile.close()
outputFile.close()

df = pd.read_csv("convertedData.csv")    #read the converted csv dat into dataframe using pandas

df = df[['text']]   #trim data set to only select the text column

df.dropna(inplace=True) #Drop rows that are missing any data

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(txt):
        txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt, re.I|re.A)
        txt = txt.lower()                                           #makes all text lower case
        txt = txt.strip()                                           #removes any trailing or leading white spaces
        tokens = nltk.word_tokenize(txt)                            #tokenize words
        clean_tokens = [t for t in tokens if t not in stop_words]   #get all the words in tokens where tokens is not a stop word
        return ' '.join(clean_tokens)

normalize_corpus = np.vectorize(normalize_document)                 #use numpy vectorize tool to normalize the entire document

norm_corpus = normalize_corpus(list(df['text']))                    #create a function to convert the normalized corpus into a dataframe

print(f'{"Total Rows in data selection: "}{len(norm_corpus)}{".  First Row of text: "}{norm_corpus[0]}')     #show results of normalized document
"""
from sklearn.feature_extraction.text import TfidfVectorizer                         #import TfidVectorizer from sklearn

tf = TfidfVectorizer(ngram_range=(1,2), min_df=2)                                    #TfidVectorizer function using ngram range of 1 to 2 words or phrases.

tfidf_matrix = tf.fit_transform(norm_corpus)                                        #Calling fit_transform fuction on the new list(norm_corpus) to create an array of words to a matrix

tfidf_matrix.shape                                                                  #Calling fuction to perform maching learning on new matrix

print(f'{"Observe the shape of the transformed matrix: "}{tfidf_matrix.shape}')     #Observing the results of the matrix to observe size

from sklearn.metrics.pairwise import cosine_similarity

doc_sim = cosine_similarity(tfidf_matrix)           #function that runs cosine similarity function on the matrix

"""

#kmeans clustering

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range = (1,2), min_df=10,max_df=.8, stop_words=stop_words)       #pass in single and double words within the listed range

cv_matrix = cv.fit_transform(norm_corpus)

cv_matrix.shape

from sklearn.cluster import KMeans

km = KMeans(n_clusters = 6, max_iter = 300, n_init = 10)

km.fit(cv_matrix)

print(f'{"Array of clusters: "}{km.labels_}')

from collections import Counter

Counter(km.labels_)

print(f'{"The Labeled Clusters 0-5, which clusters the articles by similarity.  The Cluster numbers are: "}{Counter(km.labels_)}')

df['kmeans_cluster'] = km.labels_

article_clusters = (df[['text', 'kmeans_cluster']].sort_values(by=['kmeans_cluster'], ascending=False).groupby('kmeans_cluster').head(20))

print(article_clusters)

feature_name = cv.get_feature_names()       #cv to call function to get feature names of each cluster

topn_features = 10              #create function for top feature "n" names within each cluster

ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]      #create function to order the centroids

print(ordered_centroids)


for cluster_num in range(6):        #create function to find the top n features in each cluster
    key_features = [feature_name[index]
            for index in ordered_centroids[cluster_num, :topn_features]]
    articles = article_clusters[article_clusters['kmeans_cluster']==cluster_num]['text'].values.tolist()
    print('CLUSTER #: '+ str(cluster_num+1))
    print('Key Features in each cluster: ', key_features)
    print('-'*80)

import sys
file = open('output.txt', 'a')
sys.stdout = file

print(f'{"Array of clusters: "}{km.labels_}')
print(f'{"The Labeled Clusters 0-5, which clusters the articles by similarity.  The Cluster numbers are: "}{Counter(km.labels_)}')
print(f'{"Article Clusters: "}{article_clusters}')
print(f'{"Ordered Centroids: "}{ordered_centroids}')
for cluster_num in range(6):        #create function to find the top n features in each cluster
        key_features = [feature_name[index]
            for index in ordered_centroids[cluster_num, :topn_features]]
        articles = article_clusters[article_clusters['kmeans_cluster']==cluster_num]['text'].values.tolist()
        print('CLUSTER #: '+ str(cluster_num+1))
        print('Key Features in each cluster: ', key_features)
        print('-'*80)
file.close()
