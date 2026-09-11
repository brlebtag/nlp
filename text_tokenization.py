import spacy

nlp = spacy.load('pt_core_news_sm')

text = """
Indica o tipo de dado em que o modelo foi treinado.
O modelo em português foi treinado majoritariamente com textos jornalísticos e corporativos (news),
enquanto o americano utiliza dados diversos da internet (web).
"""

doc = nlp(text.strip())

# print tokens
print([token.text for token in doc])

# print sentences
print([sent for sent in doc.sents])

# print token, gramatical class, syntatic dep
print ([(token.text, token.pos_, token.dep_) for token in doc])