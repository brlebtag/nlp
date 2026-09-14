#
# EXERCISE: Learn how to extend spaCy's NER models. Specifically, how to add new
# entity names and entity types. 
#
import spacy

# 1. Carrega o modelo existente (ex: português)
nlp = spacy.load("pt_core_news_sm")

# 2. Adiciona o EntityRuler ANTES do NER existente para ter prioridade
# (ou 'after="ner"' se quiser que o modelo estatístico tenha prioridade)
ruler = nlp.add_pipe("entity_ruler", before="ner")

# 3. Define os novos nomes (patterns) e tipos (labels)
patterns = [
    {"label": "TECNOLOGIA", "pattern": "spaCy"},
    {"label": "TECNOLOGIA", "pattern": "Python"},
    {"label": "CRIATIVO", "pattern": [{"LOWER": "inteligência"}, {"LOWER": "artificial"}]}
]

ruler.add_patterns(patterns)

# 4. Testa o modelo estendido
texto = "Eu estou aprendendo spaCy e Inteligência Artificial usando Python. Não é legal isso?"
doc = nlp(texto)

for ent in doc.ents:
    print(f"Entidade: {ent.text} -> Tipo: {ent.label_}")