"""
NLP utils
=========

"""

from functools import lru_cache

import pandas as pd
import spacy
from nesta.packages.nlp_utils.ngrammer import Ngrammer
from topsbm import TopSBM


@lru_cache()
def _nlp():
    """[summary]

    Returns:
        [type]: [description]
    """
    return spacy.load("en_core_web_sm")


def nlp(text):
    """[summary]

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
    return _nlp()(text)


def vectorise_docs(docs):
    # Process the text
    ngrammer = Ngrammer(config_filepath="/Users/jklinger/Nesta/nesta/nesta/core/config/mysqldb.config",
                        database="production")
    docs = [ngrammer.process_document(doc) for doc in docs]
    # Vectorise the docs
    vec = CountVectorizer(token_pattern=r'\S+')
    X = vec.fit_transform(['\n'.join(' '.join(sent)
                                     for sent in doc) for doc in docs])
    return X, vec


def fit_topics(X, vec, random_state=1966):
    model = TopSBM(random_state=random_state)
    Xt = model.fit_transform(X)
    topics = pd.DataFrame(model.groups_[1]['p_w_tw'],
                          index=vec.get_feature_names())
    docs = pd.DataFrame(model.groups_[1]['p_tw_d'],
                        columns=titles)
    return topics
