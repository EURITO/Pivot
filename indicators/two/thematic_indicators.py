from nesta.packages.geo_utils.country_iso_code import country_iso_code_to_name

from indicators.core.config import INDICATORS
from indicators.core.core_utils import flatten
from indicators.core.nuts_utils import get_nuts_info_lookup
from indicators.core.nlp_utils import parse_corex_topics

import pandas as pd
from datetime import datetime as dt
from functools import partial, lru_cache
from skbio.diversity.alpha import shannon


BASE_PATH = (
    "/Users/jklinger/Nesta/Pivot/indicators/two/topic_iteration/topic-model-{}-100-25"
)

BASE_KWARGS = dict(
    old_from_date="2015-01-01", old_to_date="2020-01-01", covid_start="2020-03-01"
)

DATA_PARAMS = [
    {
        "dataset": "cordis",
        "entity_type": "projects",
        "data_getter": get_cordis_projects,
        "datefield": "start_date",
        "weight_field": "funding",
        "geo_split": True,
    },
    {
        "dataset": "nih",
        "entity_type": "projects",
        "data_getter": get_nih_projects,
        "datefield": "start_date",
        "weight_field": "funding",
        "geo_split": True,
    },
    {
        "dataset": "arxiv",
        "entity_type": "articles",
        "data_getter": get_arxiv_articles,
        "datefield": "created",
        "weight_field": None,
        "geo_split": True,
    },
]


ENTITY_TYPE = {
    "cordis": "projects",
    "nih": "projects",
    "arxiv": "articles",
    "cordis-funding": "funding [€]",
    "nih-funding": "funding [$]",
}


def get_objects_and_labels(data_getter, topic_model_path):
    """Load the objects from the given data getter, match them to topics, and remove "bad" topics.
    Bad topics are defined as those which contain lots of "antitopics", "stop" topics or
    topics which have little Correlation Explanation ("fluffy" topics)

    Args:
        data_getter (func): A function returning a list of dictionaries.
        topic_model_path (str): Wherever CorEx has saved your topic model to.
    Returns:
        objects, labels (DataFrame, list): objects from the given data getter, and "good" topics from the topic model
    """
    # Load objects, topics, object-topic mapping and the total correlation of each topic
    _objects = list(data_getter())[0]
    objects = pd.DataFrame(_objects)
    topics = parse_corex_topics(topic_model_path.format("topics.txt"))
    labels = pd.read_csv(
        topic_model_path.format("labels.txt"), names=topics, index_col=0
    )
    total_correlation = pd.read_csv(
        topic_model_path.format("most_deterministic_groups.txt")
    )
    # Define fluffy, stop and antitopics
    is_fluffy = total_correlation[" NTC"].apply(lambda x: abs(x) < 0.02)
    fluffy_topics = [
        topics[itopic] for itopic in total_correlation["Group num."].loc[is_fluffy]
    ]
    non_stop_topics = pd.Series(topics, index=topics)[
        (labels.mean(axis=0) < 0.3)
    ].values.tolist()
    antitopics = [t for t in topics if t.count("~") >= 3]
    # Retain only the good topics
    labels = labels[(set(non_stop_topics) - set(antitopics)) - set(fluffy_topics)]
    return objects, labels


def sum_activity(objs, labels, date_label):
    """Extract the total activity of the provided objects, by topic in the given date range

    Args:
        objs (DataFrame): Objects over which to calculate total activity
        labels (int): CorEx's binary labels matrix, provided by CorEx.
        date_label (str): Name in the indicator.yaml config file of the date set to use
    Returns:
        activity (float): Total activity over the given time period, scaled to another time period.
    """
    from_date = INDICATORS[date_label]["from_date"]
    to_date = INDICATORS[date_label]["to_date"]
    _date = objs["created"]
    total_days = (pd.to_datetime(to_date) - pd.to_datetime(from_date)).days
    norm = days_of_covid() / total_days
    slicer = (_date > pd.to_datetime(from_date)) & (_date < pd.to_datetime(to_date))
    activity = labels[slicer].sum(axis=0).sort_values()
    return norm * activity


def thematic_diversity(objs, labels, is_covid):
    """Calculate the Shannon diversity of objects by topic, for objects tagged as covid-related and non-covid-related.


    Args:
        objs (DataFrame): Objects over which to calculate total activity
        labels (int): CorEx's binary labels matrix, provided by CorEx.
        datefield (str): The name of the date column in the dataframe.
        is_covid (Series): boolean indexer, indicating projects tagged as covid and non-covid related.
        from_date (str, default='2020-03-01'): The start of the date range to consider.
    Returns:
        diversity_covid, diversity_noncovid: Thematic diversity for covid-related and non-covid-related objects.
    """
    _date = objs["created"]
    from_date = INDICATORS["covid_dates"]["from_date"]
    in_date_range = _date > pd.to_datetime(from_date)
    return shannon(labels.loc[in_date_range & is_covid].sum(axis=0))


def covid_filterer(topic):
    """Rough rule for determining whether a topic is covid-related. Returns a boolean indexer."""
    covid_terms = ["covid", "covid-19", "coronavirus"]
    return any(term in topic for term in covid_terms)


@lru_cache()
def days_of_covid():
    """Number of days since Covid emergency declared"""
    covid_start = INDICATORS["covid_dates"]["from_date"]
    covid_end = INDICATORS["covid_dates"]["to_date"]
    return (pd.to_datetime(covid_end) - pd.to_datetime(covid_start)).days


def safe_divide(numerator, denominator):
    """In the case where there is no past activity, take 1 as an upper bound"""
    denominator.loc[denominator == 0] = 1
    return numerator / denominator


def _generate_indicators(topic_module, geo_index, weight_field):
    """Generate a suite of indicators for a given set of objects"""

    objects, topics = get_objects_and_labels(topic_module)
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
    sum_activity_all = partial(sum_activity, objects, topics)
    sum_activity_covid = partial(
        sum_activity, objects.loc[is_covid], topics.loc[is_covid]
    )
    sum_activity_noncovid = partial(
        sum_activity, objects.loc[~is_covid], topics.loc[~is_covid]
    )

    # Let's make indicators
    indicators = {}

    # Levels of activity by topic, total (2020)
    indicators["total_activity"] = sum_activity_all("covid_dates")

    # Levels of activity by topic, relative to the avg from 2015-2019
    _norm_past_activity = sum_activity_all("precovid_dates")
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


def make_indicator_description(indicator_name, topic_module):
    entity_type = topic_module.model_config["metadata"]["entity_type"]
    return INDICATORS["verbose_indicator_names"][indicator_name].format(entity_type)


def make_ctry_metadata(ctry_code):
    nuts_lookup = get_nuts_info_lookup()  # NB: lru_cached
    if ctry_code.startswith("iso_"):
        ctry_code = ctry_code[4:]
        nuts_level = 0
        nuts_name = country_iso_code_to_name(ctry_code, iso2=True)
        filename = "by-country.csv"
    else:
        nuts_level = len(ctry_code) - 1
        nuts_name = nuts_lookup[ctry_code]["nuts_name"]
        filename = f"nuts-{nuts_level}.csv"
    return ctry_code, nuts_level, nuts_name, filename


if __name__ == "__main__":

    import boto3
    from pathlib import Path
    from collections import defaultdict

    # Generate all indicators
    indicators = {}
    for module in (arxiv_topics, nih_topics, cordis_topics):
        dataset = module.model_config["dataset_label"]
        # Indicators wrt to total activity counts
        indicators[dataset] = generate_indicators(module)
        # arXiv does not have funding info
        if "funding_currency" not in module.model_config["dataset_label"]:
            continue
        # Indicators wrt to total funding
        label = f"{dataset}-funding"
        indicators[label] = generate_indicators(module, weight_field="funding")

    # Prepare and format the data, with metadata
    output = defaultdict(list)  # mapping of filepath --> [meta]data
    for ds_name, ctry_code, indc_name, topic_name, indc_value in flatten(indicators):
        if pd.isnull(indc_value) or indc_value == 0:
            continue
        indc_desc = make_indicator_description(indc_name, ds_name)
        ctry_code, nuts_level, nuts_name, filename = make_ctry_metadata(ctry_code)
        pseudofile = output[f"{ds_name}/{topic_name}/{filename}"]
        pseudofile.append(
            {
                "indicator_name": indc_name,
                "indicator_value": indc_value,
                "indicator_description": indc_desc,
                "nuts_level": nuts_level,
                "nuts_code": ctry_code,
                "nuts_name": nuts_name,
            }
        )

    # Save the data to disk and S3
    s3 = boto3.resource("s3")
    all_paths = []
    for path, data in output.items():
        df = pd.DataFrame(data)
        df = (
            df.sort_values(by=["indicator_name", "nuts_level", "nuts_code"])
            .reset_index(drop=True)
            .dropna(axis=0)
        )
        if len(df) == 0:
            continue
        Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        s3.Bucket("eurito-csv-indicators").upload_file(path, path)
        all_paths.append(
            f"https://eurito-csv-indicators.s3-eu-west-1.amazonaws.com/{path}"
        )
        del df
