#
# EXERCISE: fetch multiple topics from the 20 newsgroups
# dataset and query them using the approach we followed.
# A list of topics can be found here:
# https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
#
# If you're feeling ambitious, incorporate n-grams or
# look at how you can measure precision and recall.
#
import spacy
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = fetch_20newsgroups(categories=["sci.space", "comp.graphics", "rec.autos", "sci.med"], remove=('headers', 'footers', 'quotes'))

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

unwanted_pipes = ["ner", "parser"]

def spacy_tokenizer(doc):
    return [t.lemma_ for t in nlp(doc) if not t.is_punct and not t.is_space and t.is_alpha]

def top_k(arr, k):
    kth_largest = (k + 1) * -1
    return np.argsort(arr)[:kth_largest:-1]

vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, token_pattern=None)
features = vectorizer.fit_transform(corpus.data)
query = ["lunar orbit satellite"]
query_tfidf = vectorizer.transform(query)
cosine_similarities = cosine_similarity(features, query_tfidf).flatten()
top_related_indices = top_k(cosine_similarities, 5)
print(top_related_indices)
print("=" * 50)
print(cosine_similarities[top_related_indices])
print("=" * 50)
print("First Top match")
print(corpus.data[top_related_indices[0]])
print("=" * 50)
print("Second Top match")
print(corpus.data[top_related_indices[1]])

