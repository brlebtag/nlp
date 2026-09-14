import spacy
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = fetch_20newsgroups(categories=["sci.space"], remove=('headers', 'footers', 'quotes'))

# print(type(corpus))
# print(corpus.data)
# print(corpus.data[:2])

nlp = spacy.load("en_core_web_sm")

unwanted_pipes = ["ner", "parser"]

def spacy_tokenizer(doc):
    with nlp.disable_pipes(unwanted_pipes):
        return [t.lemma_ for t in nlp(doc) if not t.is_punct and not t.is_space and t.is_alpha]

vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
features = vectorizer.fit_transform(corpus.data)

print(len(vectorizer.get_feature_names_out()))
print("=" * 50)

query = ["lunar orbit"]
query_tfidf = vectorizer.transform(query)

cosine_similarities = cosine_similarity(features, query_tfidf).flatten()


def top_k(arr, k):
    kth_largest = (k + 1) * -1
    return np.argsort(arr)[:kth_largest:-1]

top_related_indices = top_k(cosine_similarities, 5)
print(top_related_indices)
print("=" * 50)
print(cosine_similarities[top_related_indices])
print("=" * 50)
print("Top match")
print(corpus.data[top_related_indices[0]])
print("=" * 50)
print("Second Top match")
print(corpus.data[top_related_indices[1]])


query = ["satellite"]
query_tfidf = vectorizer.transform(query)
cosine_similarities = cosine_similarity(features, query_tfidf).flatten()
top_related_indices = top_k(cosine_similarities, 5)

print("=" * 50)
print(top_related_indices)
print(cosine_similarities[top_related_indices])
print(corpus.data[top_related_indices[0]])