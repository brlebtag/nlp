import spacy
from nltk.stem.snowball import PortugueseStemmer

nlp = spacy.load('pt_core_news_sm')

text = """
O ministro Alexandre de Moraes, do Supremo Tribunal Federal (STF),
encaminhou nesta sexta-feira (11) um ofício ao presidente da Corte,
Luiz Edson Fachin, no qual cita uma "escolha seletiva" na retirada do sigilo,
por André Mendonça, dos inquéritos e procedimentos ligados ao Banco Master.
"""

doc = nlp(text.strip())

print("="*20, "part-of-speech (course-grained tags)", "="*20)
# print part-of-speech (course-grained tags)
print([(token.text, token.pos_) for token in doc])

print("="*20, "part-of-speech (fine-grained tags)", "="*20)
# print part-of-speech (fine-grained tags)
print([(token.text, token.tag_) for token in doc])

print("="*20, "entity-type", "="*20)
# print entity-type
print([(token.text, token.ent_type_) for token in doc])

print("="*20, "entity-type only", "="*20)
# print entity-type only
print([(token.text, token.ent_type_) for token in doc if token.ent_type != 0])

print("="*20, "entity-type using ents", "="*20)
# print entity-type using ents
print([(ent.text, ent.label_) for ent in doc.ents])

print("="*20, "entity-type using ents, with begin and end", "="*20)
# print entity-type using ents, with begin and end
print([(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents])

print("="*20, "dependency labels", "="*20)
# print dependency labels
print([(token.text, token.dep_) for token in doc])

print("="*20, "dependency labels with head", "="*20)
# print dependency labels with head
print([(token.text, token.dep_, token.head.text) for token in doc])

matcher = spacy.matcher.Matcher(nlp.vocab)

pattern = [
    {"TEXT": "encaminhou"},
    {"POS": "ADP", "OP": "+"},
    {"POS": "NOUN", "OP": "+"},
]

matcher.add("USER_INTENT", [pattern])

matches = matcher(doc)

print("="*20, "pattern matched  results", "="*20)
# print pattern matched  results
print("Matches", [doc[start:end].text for match_id, start, end in matches])

print("="*20, "noun phrase", "="*20)

for noun_phrase in doc.noun_chunks:
    print("phrase: {}, root head: {}".format(noun_phrase, noun_phrase.root.head))


def yodize(s: str) -> str:
    doc = nlp(s)
    for t in doc:
        if t.dep_ == "ROOT":
            # assume the sentence is subject-verb-object:
            # root is usually the verb
            # we return object-verb-subject
            seq = [doc[t.i + 1: -1].text, doc[0: t.i].text, t.text + '.']
            seq[0] = seq[0].capitalize()
            return ' '.join(seq)

print("="*20, "yoda", "="*20)

# print yoda version of the text
print(yodize("Eu gosto de comer"))