from indicators.core.config import INDICATORS
from indicators.core.indicator_utils import days_of_covid, sort_save_and_upload
from indicators.core.nlp_utils import parse_corex_topics
from indicators.two import arxiv_topics, nih_topics, cordis_topics

import pandas as pd
from pathlib import Path
from functools import partial
from skbio.diversity.alpha import shannon


LABELS_TXT = "labels.txt"
TOTCORR_TXT = "most_deterministic_groups.txt"


def parse_topics(topic_module):
    """Load the objects from the given data getter, match them to topics, and remove "bad" topics.
    Bad topics are defined as those which contain lots of "antitopics", "stop" topics or
    topics which have little Correlation Explanation ("fluffy" topics)

    Args:

    Returns:
        objects, labels (DataFrame, list): objects from the given data getter, and "good" topics from the topic model
    """
    # Load objects, topics, object-topic mapping and the total correlation of each topic
    topic_module_path = Path(topic_module.__file__).parent
    topics = parse_corex_topics(topic_module_path)
    labels = pd.read_csv(topic_module_path / LABELS_TXT, names=topics, index_col=0)
    total_corr = pd.read_csv(topic_module_path / TOTCORR_TXT)
    # Pull out parsing hyperparameters from config
    fluffy_threshold = INDICATORS["topic_parsing"]["fluffy_threshold"]
    stop_topic_threshold = INDICATORS["topic_parsing"]["stop_topic_threshold"]
    max_antitopic_count = INDICATORS["topic_parsing"]["max_antitopic_count"]
    # Define "fluffy", "stop" and "antitopics" (see docstring for definitions)
    fluffy = total_corr[" NTC"].apply(lambda x: abs(x) < fluffy_threshold)
    fluffy_topics = [topics[itopic] for itopic in total_corr["Group num."].loc[fluffy]]
    is_not_stop = labels.mean(axis=0) < stop_topic_threshold
    non_stop_topics = pd.Series(topics, index=topics)[is_not_stop].values.tolist()
    antitopics = [t for t in topics if t.count("~") > max_antitopic_count]
    # Retain only the good topics
    labels = labels[(set(non_stop_topics) - set(antitopics)) - set(fluffy_topics)]
    return labels


def sum_activity(objs, labels, date_label, indexer=None):
    """Extract the total activity of the provided objects, by topic in the given date range

    Args:
        objs (DataFrame): Objects over which to calculate total activity
        labels (int): CorEx's binary labels matrix, provided by CorEx.
        date_label (str): Name in the indicator.yaml config file of the date set to use
    Returns:
        activity (float): Total activity over the given time period, scaled to another time period.
    """
    if indexer is not None:
        objs = objs.loc[indexer]
        labels = labels.loc[indexer]
    from_date = INDICATORS[date_label]["from_date"]
    to_date = INDICATORS[date_label]["to_date"]
    _date = objs["created"]  # "created" is the name of the date field in all datasets
    total_days = (pd.to_datetime(to_date) - pd.to_datetime(from_date)).days
    norm = days_of_covid / total_days
    slicer = (_date > pd.to_datetime(from_date)) & (_date < pd.to_datetime(to_date))
    activity = labels[slicer].sum(axis=0).sort_values()
    return norm * activity


def thematic_diversity(objs, labels, is_covid):
    """
    Calculate the Shannon diversity of objects by topic, for objects tagged as
    covid-related and non-covid-related.


    Args:
        objs (DataFrame): Objects over which to calculate total activity
        labels (int): CorEx's binary labels matrix, provided by CorEx.
        datefield (str): The name of the date column in the dataframe.
        is_covid (Series): boolean indexer, indicating projects tagged as covid
                           and non-covid related.
        from_date (str, default='2020-03-01'): The start of the date range to consider.
    Returns:
        diversity_covid, diversity_noncovid: Thematic diversity for covid-related
                                             and non-covid-related objects.
    """
    _date = objs["created"]
    from_date = INDICATORS["covid_dates"]["from_date"]
    in_date_range = _date > pd.to_datetime(from_date)
    return shannon(labels.loc[in_date_range & is_covid].sum(axis=0))


def covid_filterer(topic):
    """
    Rough rule for determining whether a topic is covid-related.
    Returns a boolean indexer.
    """
    covid_terms = ["covid", "covid-19", "coronavirus"]
    return any(term in topic for term in covid_terms)


def _generate_indicators(topic_module, geo_index, weight_field):
    """Generate a suite of indicators for a given set of objects"""
    # Get objects and topics for this topic module
    object_generator = topic_module.get_objects()
    objects = pd.DataFrame(list(object_generator)[0])
    topics = parse_topics(topic_module)

    # Filter out those in this geography
    objects = objects.loc[geo_index]
    topics = topics.loc[geo_index]

    # Apply filter to data based on topic labels and date
    # 1) Work out which is the covid topic
    covid_label = list(filter(covid_filterer, topics))[0]
    # 2) Slice by being covid or non-covid related
    is_covid = topics[covid_label] == 1
    # 3) reweight by funding, if specified, instead of raw counts
    weight = 1 if weight_field is None else objects[weight_field]
    topics = topics.multiply(weight, axis=0)

    # Convenience methods for getting total activity
    sum_activity_ = partial(sum_activity, objects, topics)
    sum_activity_covid = partial(sum_activity_, indexer=is_covid)
    sum_activity_noncovid = partial(sum_activity_, indexer=~is_covid)

    # Let's make indicators
    indicators = {}

    # Levels of activity by topic, total (2020)
    indicators["total_activity"] = sum_activity_("covid_dates")

    # Levels of activity by topic, relative to the avg from 2015-2019
    _norm_past_activity = sum_activity_("precovid_dates")
    indicators["relative_activity"] = safe_divide(
        indicators["total_activity"], _norm_past_activity
    )

    # Levels of activity by topic, relative to the avg from 2015-19, covid tagged
    _total_activity = sum_activity_covid("covid_dates")
    _norm_past_activity = sum_activity_covid("precovid_dates")
    indicators["relative_activity_covid"] = safe_divide(
        _total_activity, _norm_past_activity
    )

    # Levels of activity by topic, relative to the avg from 2015-19, non-covid tagged
    _total_activity = sum_activity_noncovid("covid_dates")
    _norm_past_activity = sum_activity_noncovid("precovid_dates")
    indicators["relative_activity_noncovid"] = safe_divide(
        _total_activity, _norm_past_activity
    )

    # Overrepresentation of covid tagged compared to non-covid tagged projects
    denominator = indicators["relative_activity_noncovid"].copy()
    indicators["overrepresentation_activity"] = safe_divide(
        indicators["relative_activity_covid"], denominator
    )

    # Thematic diversity of co-occurring topics of covid and non-covid tagged projects
    indicators["thematic_diversity"] = {
        "covid-related-projects": thematic_diversity(objects, topics, is_covid),
        "non-covid-related-projects": thematic_diversity(objects, topics, ~is_covid),
    }
    return indicators


def generate_indicators(topic_module):
    indicators = {
        geo_label: _generate_indicators(topic_module, geo_index, weight_field)
        for geo_index, geo_label in topic_module.get_objects(geo_split=True)
    }
    return indicators


def make_indicators(*modules):
    # Generate all indicators
    indicators = {}
    for module in modules:
        dataset = module.model_config["dataset_label"]
        # Indicators wrt to total activity counts
        indicators[dataset] = generate_indicators(module)
        # e.g. arXiv does not have funding info
        if "funding_currency" not in module.model_config["dataset_label"]:
            continue
        # Indicators wrt to total funding
        label = f"{dataset}-funding"
        indicators[label] = generate_indicators(module, weight_field="funding")
    return indicators


if __name__ == "__main__":
    # Generate indicators in the form [dataset][geography][indicator_name][topic_name]
    indicators = make_indicators(arxiv_topics, nih_topics, cordis_topics)
    # Flatten, rejig, sort and save the data, then upload to S3
    sort_save_and_upload(indicators)
