from indicators.core.config import INDICATORS
from indicators.core.indicator_utils import (
    days_of_covid,
    sort_save_and_upload,
    safe_divide,
)
from indicators.core.nlp_utils import parse_clean_topics

import pandas as pd
from functools import partial
from skbio.diversity.alpha import shannon


def sum_activity(objs, labels, date_label, indexer=None):
    """
    Extract the total activity of the provided objects, by topic in the
    given date range

    Args:
        objs (DataFrame): Objects over which to calculate total activity
        labels (DataFrame): CorEx's binary labels matrix, provided by CorEx.
        date_label (str): Name in the indicator.yaml config file of the
                          date set to use
    Returns:
        activity (float): Total activity over the given time period,
                          scaled to another time period.
    """
    if indexer is not None:
        objs = objs.loc[indexer]
        labels = labels.loc[indexer]
    from_date = INDICATORS[date_label]["from_date"]
    to_date = INDICATORS[date_label]["to_date"]
    _date = objs["created"]  # "created" is the name of the date field in all datasets
    total_days = (pd.to_datetime(to_date) - pd.to_datetime(from_date)).days
    norm = days_of_covid / (total_days + 1)  # + 1 to be inclusive of days
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
        is_covid (Series): boolean indexer, indicating projects tagged as covid
                           and non-covid related.
    Returns:
        diversity: Thematic diversity for covid-related non-covid-related objects.
    """
    _date = objs["created"]
    from_date = INDICATORS["covid_dates"]["from_date"]
    in_date_range = _date > pd.to_datetime(from_date)
    return shannon(labels.loc[in_date_range & is_covid].sum(axis=0))


def covid_filterer(topic):
    """
    Fixed rule for determining whether a topic is covid-related.
    Returns a boolean indexer.
    """
    covid_terms = ["covid", "covid-19", "coronavirus"]
    return any(term in topic for term in covid_terms)


def covid_topic_indexer(topics):
    """
    Returns an indexer over topic labels, indicating which are covid-related.
    """
    covid_topics = list(filter(covid_filterer, topics))[0]
    return topics[covid_topics] == 1


def relative_activity(activity_summer):
    """
    Calculate total activity during "covid times" relative to the expectation
    from "precovid times"
    """
    total_activity = activity_summer("covid_dates")
    norm_past_activity = activity_summer("precovid_dates")
    return safe_divide(total_activity, norm_past_activity)


def get_objects_and_topics(topic_module, geo_index, weight_field):
    # Get objects and topics for this topic module
    object_generator = topic_module.get_objects()
    objects = pd.DataFrame(list(object_generator)[0])
    topics = parse_clean_topics(topic_module)

    # Filter out those in this geography
    objects = objects.loc[geo_index]
    topics = topics.loc[geo_index]

    # Reweight by funding, if specified, instead of raw counts
    weight = 1 if weight_field is None else objects[weight_field]
    topics = topics.multiply(weight, axis=0)
    return objects, topics


def generate_indicators(topic_module, geo_index, weight_field):
    """Generate a suite of indicators for a given set of objects"""
    objects, topics = get_objects_and_topics(topic_module, geo_index, weight_field)
    is_covid = covid_topic_indexer(topics)

    # Convenience methods for making indicators
    sum_activity_ = partial(sum_activity, objects, topics)
    total_activity = partial(sum_activity_, "covid_dates")
    sum_activity_covid = partial(sum_activity_, indexer=is_covid)
    sum_activity_noncovid = partial(sum_activity_, indexer=~is_covid)
    diversity = partial(thematic_diversity, objects, topics)

    # Let's make indicators
    indicators = {
        # Levels of activity by topic, total (2020)
        "total_activity": total_activity(),
        # Levels of activity by topic, relative to the avg from 2015-2019
        "relative_activity": relative_activity(sum_activity_),
        "relative_activity_covid": relative_activity(sum_activity_covid),
        "relative_activity_noncovid": relative_activity(sum_activity_noncovid),
        # Thematic diversity of co-occurring topics of covid and non-covid projects
        "thematic_diversity": {
            "covid-related-projects": diversity(is_covid),
            "non-covid-related-projects": diversity(~is_covid),
        },
    }

    # Overrepresentation of covid tagged compared to non-covid tagged projects
    indicators["overrepresentation_activity"] = safe_divide(
        indicators["relative_activity_covid"], indicators["relative_activity_noncovid"]
    )

    # Convert all to dict
    return {k: dict(v) for k, v in indicators.items()}


def indicators_by_geo(topic_module, weight_field=None):
    """Generate indicators for all available geographic splits of this dataset"""
    indicators = {
        geo_name: generate_indicators(topic_module, geo_idx, weight_field)
        for geo_idx, geo_name in topic_module.get_objects(geo_split=True)
    }
    return indicators


def make_indicators(*modules):
    """
    Iterate over all dataset modules to generate indicators in terms of
    total counts and total funding, split by all available geographic levels
    """
    # Generate all indicators
    indicators = {}
    for module in modules:
        dataset = module.model_config["dataset_label"]
        # Indicators wrt to total activity counts
        indicators[dataset] = indicators_by_geo(module)
        # e.g. arXiv does not have funding info
        if "funding_currency" not in module.model_config["dataset_label"]:
            continue
        # Indicators wrt to total funding
        label = f"{dataset}-funding"
        indicators[label] = indicators_by_geo(module, weight_field="funding")
    return indicators


if __name__ == "__main__":
    from indicators.two import arxiv_topics, nih_topics, cordis_topics

    # Indicators in the form [dataset][geo][indicator_name][topic_name]
    indicators = make_indicators(arxiv_topics, nih_topics, cordis_topics)
    # Flatten, sort, save locally, then upload to S3
    sort_save_and_upload(indicators)
