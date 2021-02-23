"""
NLP utils
=========

"""

import os
from corextopic import vis_topic as vt
from corextopic import corextopic as ct

import pickle
import pandas as pd
from nesta.packages.nlp_utils.ngrammer import Ngrammer
from sklearn.feature_extraction.text import CountVectorizer
from indicators.core.config import mysql_config


def join_text(*args):
    """Concatenate all text arguments together, ignore None values.

    Args:
        args (tuple): Any sequence of str or None values.
    Returns:
        joined: All text values concatenated together with a single space token.
    """
    return ' '.join(filter(None, args))


def join_doc(doc, extra_stops=[]):
    """Join and conservatively lemmatise a term-wise document

    Args:
        doc (list of list): Document of terms to be joined and lemmatised.

    Returns:
        _doc: Joined and lemmatised documents
    """
    term_filterer = lambda term: term not in extra_stops
    sent_joiner = lambda sent: join_text(*filter(term_filterer, sent))
    joined_sents = map(sent_joiner, doc)
    joined_doc = '\n'.join(joined_sents)
    return joined_doc


def vectorise_docs(docs, min_df=10, max_df=0.95, extra_stops=[]):
    """Impute n-grams from wiktionary and then process using a standard
    count vectoriser.

    Args:
        docs (list): List of text documents to process.

    Returns:
        (doc_vectors, feature_names): Vectorised documents and a list of feature names.
    """
    # Process the text
    ngrammer = Ngrammer(config_filepath=mysql_config, database="production")
    docs = [ngrammer.process_document(doc) for doc in docs]
    # Join and conservatively lemmatise
    docs = map(lambda doc: join_doc(doc, extra_stops), docs)
    # Vectorise the docs
    vec = CountVectorizer(min_df=min_df, max_df=max_df)
    doc_vectors = vec.fit_transform(docs)
    return doc_vectors, vec.get_feature_names()

# %%


def fit_topics(dataset_label, doc_vectors, feature_names, titles, n_topics,
               anchors, anchor_strength=10, max_iter=25):
    """[summary]

    Args:
        dataset_label ([type]): [description]
        doc_vectors ([type]): [description]
        feature_names ([type]): [description]
        titles ([type]): [description]
        n_topics ([type]): [description]
        anchors ([type]): [description]
        anchor_strength (int, optional): [description]. Defaults to 10.
        max_iter (int, optional): [description]. Defaults to 25.

    Returns:
        [type]: [description]
    """
    kwargs = dict(X=doc_vectors, words=feature_names, docs=titles,
                  anchors=anchors, anchor_strength=anchor_strength)
    label = f'topic-model-{dataset_label}-{n_topics}-{max_iter}'

    topic_model = ct.Corex(max_iter=max_iter, n_hidden=n_topics)
    topic_model.fit(**kwargs)
    vt.vis_rep(topic_model, column_label=feature_names, prefix=label)
    with open(label + '/model.pickle', 'wb') as f:
        pickle.dump(topic_model, f)
    os.remove(label + '/cont_labels.txt')  # Very large file that we don't use
    return topic_model
# %%


def parse_topic(line, n_most=5):
    """[summary]

    Args:
        line ([type]): [description]
        n_most (int, optional): [description]. Defaults to 5.

    Returns:
        [type]: [description]
    """
    _, topic_content = line.split(":")
    topic_content = ' '.join(topic_content.split(",")[:-n_most])
    return topic_content


def parse_corex_topics(path):
    """[summary]

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    topics = []
    with open(path) as f:
        for line in f.readlines():
            topic_content = parse_topic(line)
            topics.append(topic_content)
    return topics