#
# EXERCISE: Create a spacy_tokenizer callback which takes a string and returns
# a list of tokens (each token's text) with punctuation filtered out.
#

import spacy
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


corpus = [
  "Os alunos utilizam seus celulares com GPS para tirar fotos aéreas de um terreno, a fim de localizar pontos de risco específicos, como pilhas de lixo.",
  "Os adolescentes estão entusiasmados em fazer fotografias aéreas para estudar o seu bairro.",
  "A fotografia aérea é uma excelente maneira de identificar características do terreno que não são visíveis do solo, como o contorno de lagos ou o curso de rios.",
  "Nos primórdios das câmeras DSLR digitais, a Canon era praticamente a líder incontestável em tecnologia de sensores de imagem CMOS.",
  "O presidente sírio, Bashar al-Assad, diz aos EUA que eles 'pagarão o preço' se atacarem a Síria."
]

nlp = spacy.load('pt_core_news_sm')

def spacy_tokenizer(doc):
    return [t.text for t in nlp(doc) if not t.is_punct]

#
# EXERCISE: Initialize a CountVectorizer object and set it to use
# your spacy_tokenizer with lower-casing off and to create a binary BOW.
#

# Instantiate a CountVectorizer object called 'vectorizer'.


# Create a binary BOW from the corpus using your CountVectorizer.
vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, lowercase=False, binary=True)
bow = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print("="*20)
print(vectorizer.vocabulary_)

#
# The string below is a whole paragraph. We want to create another
# binary BOW but using the vocabulary of our *current* CountVectorizer. This means
# that words in this paragraph which AREN'T already in the vocabulary won't be
# represented. This is to illustrate how BOW can't handle out-of-vocabulary words
# unless you rebuild your whole vocabulary. Still, we'll see that if there's
# enough overlapping vocabulary, some similarity can still be picked up.
#
# Note that we call 'transform' only instead of 'fit_transform' because the
# fit step (i.e. vocabulary build) is already done and we don't want to re-fit here.
#
s= ["Adolescentes fazem fotos aéreas de seu bairro usando câmeras digitais acomodadas em garrafas velhas, que são içadas por pipas — um brinquedo comum entre as crianças que vivem nas favelas. Em seguida, eles utilizam smartphones com GPS para fotografar pontos de risco específicos, como acúmulos de lixo, que podem se tornar criadouros de mosquitos transmissores da dengue."]
new_bow = vectorizer.transform(s)
print(new_bow)

#
# EXERCISE: using the pairwise cosine_similarity method from sklearn,
# calculate the similarities between each document from the corpus against
# this new document (new_bow). HINT: You can pass two parameters to
# cosine_similarity in this case. See the docs:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html#scipy.spatial.distance.cosine
#
# Which document is the most similar? Which is the least similar? Do the results make sense
# based on what you see?
#
print("="*20)
print(cosine_similarity(bow, new_bow))

#
# EXERCISE: Implement your own cosine similarity method using numpy.
# It should take two numpy arrays and output the similarity metric.
# HINTS:
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
#
# Verify the similarity between the first document in the corpus and the
# paragraph is the same as the one you got from using pairwise cosine_similarity.
#
import numpy as np
def cos_sim(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))