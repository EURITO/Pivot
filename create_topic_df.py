import os
import pandas as pd
import joblib

import logging
import yaml


def post_process_model(model, top_level):
    """Function to post-process the outputs of a hierarchical topic model
    _____
    Args:
      model:      A hsbm topic model
      top_level:  The level of resolution at which we want to extract topics
    _____
    Returns:
      A topic mix df with topics and weights by document
    """
    # Extract the word mix (word components of each topic)
    logging.info("Creating topic names")
    word_mix = model.topics(l=top_level)

    # Create tidier names
    topic_name_lookup = {
        key: "_".join([x[0] for x in values[:5]]) for key, values in word_mix.items()
    }
    topic_names = list(topic_name_lookup.values())

    # Extract the topic mix df
    logging.info("Extracting topics")
    topic_mix_ = pd.DataFrame(
        model.get_groups(l=top_level)["p_tw_d"].T,
        columns=topic_names,
        index=model.documents,
    )

    return topic_mix_


def filter_topics(topic_df, presence_thr, prevalence_thr):
    """Filter uninformative ("stop") topics
    Args:
        top_df (df): topics
        presence_thr (int): threshold to detect topic in article
        prevalence_thr (int): threshold to exclude topic from corpus
    Returns:
        Filtered df
    """
    # Remove highly uninformative / generic topics

    topic_prevalence = (
        topic_df
        .applymap(lambda x: x > presence_thr)
        .mean()
        .sort_values(ascending=False)
    )


    # Filter topics
    filter_topics = topic_prevalence.index[topic_prevalence > prevalence_thr].tolist()

    # We also remove short topics (with less than two ngrams)
    filter_topics = filter_topics + [
        x for x in topic_prevalence.index if len(x.split("_")) <= 2
    ]

    topic_df_filt = topic_df.drop(filter_topics, axis=1)

    return topic_df_filt, filter_topics