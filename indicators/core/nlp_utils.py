"""
NLP utils
=========

"""

import os
from pathlib import Path
import pickle
from functools import lru_cache

from corextopic import vis_topic as vt
from corextopic import corextopic as ct
import pandas as pd

from nesta.packages.nlp_utils.ngrammer import Ngrammer
from sklearn.feature_extraction.text import CountVectorizer
from indicators.core.config import MYSQLDB_PATH, INDICATORS
from indicators.core.core_utils import object_getter

CONFIG = INDICATORS["topic_parsing"]  # topic parsing config


def join_text(*args):
    """Concatenate all text arguments together, ignore None values.

    Args:
      *args: All str arguments to be joined

    Returns:
      joined: Single string representing the text
    """
    return " ".join(filter(None, args))


def join_and_filter_sent(sent, extra_stops):
    """Concatenate list, ignore stop terms.

    Args:
      sent (list): List of terms representing a sentence
      extra_stops (list): Stop terms to be omitted from the sentence

    Returns:
      joined: Single string representing the sentence
    """
    _filter = filter(lambda term: term not in extra_stops, sent)
    return join_text(*_filter)


def join_doc(doc, extra_stops=[]):
    """Join a term-wise document (list of list of terms)

    Args:
      doc (list of list): Document of terms to be joined.
      extra_stops: Extra stop terms to be removed from the document (Default value = [])

    Returns:
      _doc: Joined and lemmatised documents
    """

    joined_sents = map(lambda doc: join_and_filter_sent(doc, extra_stops), doc)
    joined_doc = "\n".join(joined_sents)
    return joined_doc


def vectorise_docs(docs, min_df=10, max_df=0.95, extra_stops=[]):
    """Impute n-grams from wiktionary and then process using a standard
    count vectoriser.

    Args:
      docs(list): List of text documents to process.
      min_df: (Default value = 10)
      max_df: (Default value = 0.95)
      extra_stops: (Default value = [])

    Returns:
      doc_vectors: Vectorised documents and a list of feature names.

    """
    # Process the text
    ngrammer = Ngrammer(config_filepath=MYSQLDB_PATH, database="production")
    docs = [ngrammer.process_document(doc) for doc in docs]
    # Join and conservatively lemmatise
    docs = map(lambda doc: join_doc(doc, extra_stops), docs)
    # Vectorise the docs
    vec = CountVectorizer(min_df=min_df, max_df=max_df)
    doc_vectors = vec.fit_transform(docs)
    return doc_vectors, vec.get_feature_names()


def make_model_label(dataset_label, n_topics, max_iter, **kwargs):
    """Standardised label for datasets from model config"""
    return f"topic-model-{dataset_label}-{n_topics}-{max_iter}"


def fit_topics(
    dataset_label,
    doc_vectors,
    feature_names,
    titles,
    n_topics,
    anchors,
    anchor_strength=10,
    max_iter=25,
):
    """Apply Corex topic modelling to a set of document vectors,
    and save the model and output to disk.

    Args:
      dataset_label(str): Name of this dataset, for labelling the output files
      doc_vectors(np.array): Count (or equivalent) vector repr of the documents.
      feature_names(list): Names of each feature in the doc_vectors
      titles(list): Name of each document in doc_vectors
      n_topics(int): Number of topics for Corex to generate
      anchors(list of list): Corex anchor terms
      anchor_strength(int, optional): Corex anchor strength multiplier. Defaults to 10.
      max_iter(int, optional): Number of model iterations. Defaults to 25.

    Returns:
      topic_model: trained Corex topic model
    """
    topic_model = ct.Corex(max_iter=max_iter, n_hidden=n_topics)
    topic_model.fit(
        X=doc_vectors,
        words=feature_names,
        docs=titles,
        anchors=anchors,
        anchor_strength=anchor_strength,
    )
    # Use Corex tools for writing the data to the local directory
    label = make_model_label(dataset_label, n_topics, max_iter)
    vt.vis_rep(topic_model, column_label=feature_names, prefix=label)
    os.remove(f"{label}/cont_labels.txt")  # Very large file that we don't use
    # pickle model for later use
    with open(f"{label}/model.pickle", "wb") as f:
        pickle.dump(topic_model, f)
    return topic_model


def parse_topic(raw_topic):
    """Convert a verbose Corex topic into a concise human-readable form.

    Args:
      raw_topic(str): Verbose Corex topic

    Returns:
      topic_content(str): Concise human-readable Corex topic
    """
    n_most = CONFIG["terms_in_topics"]
    _, topic_content = raw_topic.split(":")
    topic_content = " ".join(topic_content.split(",")[:n_most])
    return topic_content


@lru_cache()
def parse_corex_topics(topic_module):
    """Parse concise human-readable topics from the output
    of a Corex topic model.

    Args:
      path(path-like): Path to the Corex topic model output
      n_most(int): (Default value = 5)

    Returns:
      topics(list): List of concise human-readable topics
    """
    corex_paths = parse_corex_paths(topic_module)
    topics = []
    with open(corex_paths["topics"]) as f:
        for raw_topic in f.readlines():
            topic_content = parse_topic(raw_topic)
            topics.append(topic_content)
    return topics


def fit_topic_model(topic_module):
    """Fit topics based on hyperparameters specified in the model config.

    Args:
        model_config (dict): additional arguments for `fit_topics`
        object_getter (function): Function for retrieving objects to be fitted.

    Returns:
        objects, topic_model: List of objects (articles or projects),
                              and a trained topic model
    """
    objs = next(object_getter(topic_module))  # only one value (a list), so use next
    texts = [obj["text"] for obj in objs]
    titles = [obj["title"] for obj in objs]
    # Don't need the metadata for topic modelling
    topic_module.model_config.pop("metadata")
    # Prepare the data and fit the model
    doc_vectors, feature_names = vectorise_docs(texts)
    topic_model = fit_topics(
        titles=titles,
        doc_vectors=doc_vectors,
        feature_names=feature_names,
        **topic_module.model_config,
    )
    return objs, topic_model


@lru_cache()
def parse_corex_paths(topic_module):
    """Get a lookup to all of CorEx's .txt output paths"""
    label = make_model_label(**topic_module.model_config)
    top_dir = Path(topic_module.__file__).parent
    return {
        fname.stem: fname
        for fname in (top_dir / label).iterdir()
        if fname.suffix == ".txt"
    }


@lru_cache()
def get_corex_labels(topic_module):
    """Retrieve CoreX topic labels from output of a CorEx run"""
    corex_paths = parse_corex_paths(topic_module)
    topics = parse_corex_topics(topic_module)
    return pd.read_csv(corex_paths["labels"], names=topics, index_col=0)


def get_non_stop_topics(topic_module):
    """
    Retrieve CoreX topics and define topics as being "stop" if they occur in
    more than `stop_topic_threshold` fraction of all documents. Then return the
    set of all topic labels which are not stops.
    """
    topics = parse_corex_topics(topic_module)
    labels = get_corex_labels(topic_module)
    is_not_stop = labels.mean(axis=0) < CONFIG["stop_topic_threshold"]
    non_stop_topics = pd.Series(topics, index=topics)[is_not_stop].values.tolist()
    return set(non_stop_topics)


def get_antitopics(topic_module):
    """
    Retrieve CoreX topics and define topics as being "antitopics" if they contain
    a large number (`max_antitopic_count`) of terms which are anti-correlated with
    a topic occurence. That isn't to say that these are really "bad" topics,
    but rather they are very hard to intepret. For example, "~chicken ~building people"
    would imply a topic in which the terms "chicken" or "building" don't appear, and
    so the "physical" interpretation is difficult.
    """
    topics = parse_corex_topics(topic_module)
    return set(t for t in topics if t.count("~") > CONFIG["max_antitopic_count"])


def get_fluffy_topics(topic_module):
    """
    Retrieve CoreX topics and define topics as being "fluffy" if they explain little
    total correlation. This corresponds to the `NTC` variable in the
    `most_deterministic_groups` output, and is filtered against the a
    `fluffy_threshold` config variable. These could be interpretted as "noisy"
    topics.
    """
    corex_paths = parse_corex_paths(topic_module)
    topics = parse_corex_topics(topic_module)
    total_corr = pd.read_csv(corex_paths["most_deterministic_groups"])
    fluffy = total_corr[" NTC"].apply(lambda x: abs(x) < CONFIG["fluffy_threshold"])
    fluffy_topics = [topics[itopic] for itopic in total_corr["Group num."].loc[fluffy]]
    return set(fluffy_topics)


def parse_clean_topics(topic_module):
    """
    Retrieve all CorEx topics and filter for "nice" topics.

    Args:
        topic_module (module): A topic module, e.g. arxiv_topics
    Returns:
        labels (pd.DataFrame): Binary labels for each document in the model with
                               only "nice" topics considered.
    """
    fluffy_topics = get_fluffy_topics(topic_module)
    antitopics = get_antitopics(topic_module)
    non_stop_topics = get_non_stop_topics(topic_module)
    labels = get_corex_labels(topic_module)
    return labels[(non_stop_topics - antitopics) - fluffy_topics]
