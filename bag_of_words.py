import spacy
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
  "Red Bull drops hint on F1 engine.",
  "Honda exits F1, leaving F1 partner Red Bull.",
  "Hamilton eyes record eighth F1 title.",
  "Aston Martin announces sponsor."
]

vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(corpus)
# vocabulary = features
print(vectorizer.get_feature_names_out())
print("="*20)
# vocabulary's mappings
print(vectorizer.vocabulary_)


print("="*50)
corpus = [
    "Red Bull dá pista sobre motor de F1.",
    "Honda deixa a F1, separando-se da parceira Red Bull",
    "Hamilton mira o recorde de oito títulos na F1.",
    "Aston Martin anuncia patrocinador."
]

nlp = spacy.load("pt_core_news_sm")

def spacy_tokenizer(doc):
    return [t.text for t in nlp(doc) if not t.is_punct]

vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, token_pattern=None, lowercase=False, binary=True)
bow = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print("="*20)
print(vectorizer.vocabulary_)

print("="*50)

print("Dense representation")
print(bow.toarray())
print()
print("Indexing and slicing")
print(bow[0])
print()
print(bow[0:2])

# cosine similarity
# spatial.distance.cosine = cosine distance
# cosine similarity = 1 - (consine distance)
doc1_vs_doc2 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[1].toarray()[0])
doc1_vs_doc3 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[2].toarray()[0])
doc1_vs_doc4 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[3].toarray()[0])

print("="*50)
print(corpus)
print()
print(f"Doc 1 vs Doc 2: {doc1_vs_doc2}")
print(f"Doc 1 vs Doc 3: {doc1_vs_doc3}")
print(f"Doc 1 vs Doc 3: {doc1_vs_doc4}")

print("="*50)
print(cosine_similarity(bow))

# 1-gram and 2-gram
print("="*50)
vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, token_pattern=None, lowercase=False, binary=True, ngram_range=(1, 2))
bigrams = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print("Number of features: {}".format(len(vectorizer.get_feature_names_out())))
print(vectorizer.vocabulary_)

# 2-grams only
print("="*50)
vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, token_pattern=None, lowercase=False, binary=True, ngram_range=(2, 2))
bigrams = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print("Number of features: {}".format(len(vectorizer.get_feature_names_out())))
print(vectorizer.vocabulary_)

