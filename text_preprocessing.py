import spacy

nlp = spacy.load('pt_core_news_sm')

text = """
Indica o tipo de dado em que o modelo foi treinado.
O modelo em português foi treinado majoritariamente com textos jornalísticos e corporativos (news),
enquanto o americano utiliza dados diversos da internet (web).
"""

doc = nlp(text.strip())

# apply case-folding
print([token.lower_ for token in doc])