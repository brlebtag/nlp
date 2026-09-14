import spacy
from nltk.stem.snowball import PortugueseStemmer

nlp = spacy.load('pt_core_news_sm')

text = """
Indica o tipo de dado em que o modelo foi treinado.
O modelo em português foi treinado majoritariamente com textos jornalísticos e corporativos (news),
enquanto o americano utiliza dados diversos da internet (web).
"""

doc = nlp(text.strip())

print("="*20, "case-folding", "="*20)
# apply case-folding
print([token.lower_ for token in doc])

print("="*20, "apply case-folding after the first token in the sentence", "="*20)
# apply case-folding after the first token in the sentence.
print([token.lower_ if not token.is_sent_start else token for token in doc])

print("="*20, "total stop_words list", "="*20)
# print num stop_words
print(len(nlp.Defaults.stop_words))

print("="*20, "stop words list", "="*20)
# print stop words list
print(nlp.Defaults.stop_words)

print("="*20, "non-stop words", "="*20)
# print only non-stop words
print([token for token in doc if not token.is_stop])

print("="*20, "lemmatization", "="*20)
# print lemmatization
print([(token, token.lemma_) for token in doc])

print("="*20, "stemming", "="*20)
# print stemming
stemmer = PortugueseStemmer()
print([(token.text, stemmer.stem(token.text)) for token in doc])
