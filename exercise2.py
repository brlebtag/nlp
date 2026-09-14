#
# EXERCISE: using doc.ents, identify and print the dates in this sentence.
# Expected output: ['Feb 13th', 'Feb 24th']
#
import spacy

# 1. Carrega o modelo em português (recomenda-se o 'md' ou 'lg' para maior precisão com datas)
nlp = spacy.load("pt_core_news_md")

# 2. Processa a frase desejada
frase = "Estaremos em Osaka no dia 13 de fevereiro e partiremos no dia 24 de fevereiro."
doc = nlp(frase)

# 3. Itera sobre doc.ents filtrando pelo tipo de entidade 'DATE'
print("--- Datas encontradas ---")
for ent in doc.ents:
    if ent.label_ == "DATE":
        print(f"Data: {ent.text} (Posição: {ent.start_char} a {ent.end_char})")