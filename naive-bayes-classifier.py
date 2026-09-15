import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import fetch_20newsgroups
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# training_corpus = fetch_20newsgroups(subset='train')

# print('Training data size: {}'.format(len(training_corpus.data)))

# print("=" * 50)
# print(training_corpus.target_names)
# print("=" * 50)
# print(training_corpus.target)
# print("=" * 50)
# print(training_corpus.data[0])
# print("=" * 50)

# first_doc_label = training_corpus.target[0]
# print("Label for this post: {}".format(first_doc_label ))
# print("Corresponding topic: {}".format(training_corpus.target_names[first_doc_label]))
# bins, counts = np.unique(training_corpus.target, return_counts=True)
# freq_series = pd.Series(counts/len(training_corpus.data))
# plt.figure(figsize=(12, 8))
# ax = freq_series.plot(kind='bar')
# ax.set_xticklabels(bins, rotation=0)
# plt.show()

# train_data, val_data, train_labels, val_labels = train_test_split(training_corpus.data, training_corpus.target, train_size=0.8, random_state=1)
# print('Training data size: {}'.format(len(train_data)))
# print('Validation data size: {}'.format(len(val_data)))
# print("=" * 50)

# nlp = spacy.blank('en')

# print(nlp.pipe_names)

# print("=" * 50)

# def spacy_tokenizer(doc):
#     return [t.text for t in nlp(doc) if not t.is_punct and not t.is_space and t.is_alpha]

# vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, token_pattern=None)
# train_feature_vects = vectorizer.fit_transform(train_data)
nb_classifier = MultinomialNB()
# nb_classifier.fit(train_feature_vects, train_labels)
# print(nb_classifier.get_params())
# train_preds = nb_classifier.predict(train_feature_vects)
# print('F1 score on initial training set: {}'.format(metrics.f1_score(train_labels, train_preds, average='macro')))


# print("=" * 50)
filtered_training_corpus = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
train_data, val_data, train_labels, val_labels = train_test_split(filtered_training_corpus.data, filtered_training_corpus.target, train_size=0.8, random_state=1)
# train_feature_vects = vectorizer.fit_transform(train_data)
# nb_classifier.fit(train_feature_vects, train_labels)
# train_preds = nb_classifier.predict(train_feature_vects)
# print('F1 score on filtered validation set: {}'.format(metrics.f1_score(train_labels, train_preds, average='macro')))

# print("=" * 50)

# val_feature_vects = vectorizer.transform(val_data)
# val_preds = nb_classifier.predict(val_feature_vects)
# print('F1 score on filtered validation set: {}'.format(metrics.f1_score(val_labels, val_preds, average='macro')))

# print("=" * 50)

# # Set the size of the plot.
# fig, ax = plt.subplots(figsize=(15, 15))

# # Create the confusion matrix. 
# disp = ConfusionMatrixDisplay.from_estimator(nb_classifier,
#     val_feature_vects, val_labels,
#     normalize='true',
#     display_labels=filtered_training_corpus.target_names,
#     xticks_rotation='vertical',
#     ax=ax,
#     values_format='.2f',
#     include_values=True
# )
# plt.tight_layout()
# plt.show()

# print(metrics.classification_report(val_labels, val_preds, target_names=filtered_training_corpus.target_names))

# print("=" * 50)

# print('Training data size: {}'.format(len(train_data)))
# print('Number of training features: {}'.format(len(train_feature_vects[0].toarray().flatten())))

# print("=" * 50)

unwanted_pipes = ['ner', 'parser']

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def spacy_tokenizer(doc):
    return [t.lemma_ for t in nlp(doc) if not t.is_punct and not t.is_space and not t.is_stop and t.is_alpha]

vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, token_pattern=None)
train_feature_vects = vectorizer.fit_transform(train_data)

print('Number of training features: {}'.format(len(train_feature_vects[0].toarray().flatten())))

nb_classifier.fit(train_feature_vects, train_labels)
train_preds = nb_classifier.predict(train_feature_vects)
print('Training F1 score with fewer features: {}'.format(metrics.f1_score(train_labels, train_preds, average='macro')))

print("=" * 50)

val_feature_vects = vectorizer.transform(val_data)
val_preds = nb_classifier.predict(val_feature_vects)
print('Validation F1 score with fewer features: {}'.format(metrics.f1_score(val_labels, val_preds, average='macro')))

# fig, ax = plt.subplots(figsize=(15, 15))
# disp = ConfusionMatrixDisplay.from_estimator(nb_classifier, val_feature_vects, val_labels, normalize='true', display_labels=filtered_training_corpus.target_names, xticks_rotation='vertical', ax=ax)
# plt.tight_layout()
# plt.show()
# print(metrics.classification_report(val_labels, val_preds, target_names=filtered_training_corpus.target_names))

params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0,],}

multinomial_nb_grid = model_selection.GridSearchCV(MultinomialNB(), param_grid=params, scoring='f1_macro', n_jobs=-1, cv=5, verbose=5)
multinomial_nb_grid.fit(train_feature_vects, train_labels)
print('Best parameter value(s): {}'.format(multinomial_nb_grid.best_params_))

best_nb_classifier = multinomial_nb_grid.best_estimator_
val_preds = best_nb_classifier.predict(val_feature_vects)
print('Validation F1 score with fewer features: {}'.format(metrics.f1_score(val_labels, val_preds, average='macro')))

fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay.from_estimator(best_nb_classifier,
    val_feature_vects, val_labels,
    normalize='true',
    display_labels=filtered_training_corpus.target_names,
    xticks_rotation='vertical',
    ax=ax
)
plt.tight_layout()
plt.show()
print(metrics.classification_report(val_labels, val_preds, target_names=filtered_training_corpus.target_names))