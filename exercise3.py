#
# EXERCISE: Read about spaCy's PhraseMatcher
# https://spacy.io/usage/rule-based-matching#phrasematcher
#
# Using the PhraseMatcher, find the start and end index of all occurrences 
# of 'Caesar Augustus' and 'Roman Empire' (case-insensitive).
#
# Expected output: [(0, 2), (15, 17)]
#
import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load("pt_core_news_sm")
matcher = PhraseMatcher(nlp.vocab)
terms = ["Caesar Augustus", "Roman Empire"]
patterns = [nlp.make_doc(text) for text in terms]

matcher.add("TerminologyList", patterns)

s = "Caesar Augustus was the founder of the Roman Principate (the first phase of the Roman Empire)."
doc = nlp(s)

matches = matcher(doc)

pairs = [(start, end) for match_id, start, end in matches]
print(pairs)

