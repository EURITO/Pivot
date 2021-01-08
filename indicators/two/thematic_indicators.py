from nesta.packages.geo_utils.country_iso_code import country_iso_code_to_name
from nih_topics import get_nih_projects
from cordis_topics import get_cordis_projects
from crunchbase_topics import get_crunchbase_orgs
from arxiv_topics import get_arxiv_articles

from indicators.core.nuts_utils import get_nuts_info_lookup
from indicators.core.nlp_utils import parse_corex_topics

import pandas as pd
from datetime import datetime as dt
from functools import partial
from skbio.diversity.alpha import shannon

VERBOSE_INDICATOR_NAMES = {
                            "total_activity": 'Total activity ({}) since March 2020',
                            "relative_activity": 'Activity ({}) since March 2020, relative to the expectation from 2015-2019',
                            "relative_activity_covid": 'Covid-related activity ({}) since March 2020, relative to the expectation from 2015-2019',
                            "relative_activity_noncovid": 'Non-covid-related activity ({}) since March 2020, relative to the expectation from 2015-2019',
                            "overrepresentation_activity": 'Over-representation of covid-related activity ({}) since March 2020, compared to non-covid-related projects',
                            "thematic_diversity": "Shannon diversity of {}"
                          }

BASE_PATH = "/Users/jklinger/Nesta/Pivot/indicators/two/topic_iteration/topic-model-{}-100-25"

BASE_KWARGS = dict(old_from_date='2015-01-01', old_to_date='2020-01-01', covid_start='2020-03-01')

DATA_PARAMS = [{'dataset': 'cordis', 'entity_type': 'projects', 'data_getter': get_cordis_projects, 'datefield': 'start_date', 'weight_field':'funding', 'geo_split': True},
               {'dataset': 'nih', 'entity_type': 'projects', 'data_getter': get_nih_projects, 'datefield': 'start_date', 'weight_field':'funding', 'geo_split': True},
               {'dataset': 'arxiv', 'entity_type': 'articles', 'data_getter': get_arxiv_articles, 'datefield': 'created', 'weight_field': None, 'geo_split': True}]


ENTITY_TYPE = {'cordis': 'projects',
               'nih': 'projects',
               'arxiv': 'articles',
               'cordis-funding': 'funding [€]',
               'nih-funding': 'funding [$]'}


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
    labels = pd.read_csv(topic_model_path.format("labels.txt"), names=topics, index_col=0)
    total_correlation = pd.read_csv(topic_model_path.format("most_deterministic_groups.txt"))
    # Define fluffy, stop and antitopics
    is_fluffy = total_correlation[" NTC"].apply(lambda x: abs(x) < 0.02)
    fluffy_topics = [topics[itopic] for itopic in total_correlation['Group num.'].loc[is_fluffy]]
    non_stop_topics = pd.Series(topics,index=topics)[(labels.mean(axis=0) < 0.3)].values.tolist()
    antitopics = [t for t in topics if t.count("~") >= 3]
    # Retain only the good topics
    labels = labels[(set(non_stop_topics) - set(antitopics)) - set(fluffy_topics)]
    return objects, labels


def total_activity_by_topic_in_date_range(objs, labels, datefield, norm_days, from_date='2020-03-01', to_date=dt.now()):
    """Extract the total activity of the provided objects, by topic in the given date range

    Args:
        objs (DataFrame): Objects over which to calculate total activity
        labels (int): CorEx's binary labels matrix, provided by CorEx.
        datefield (str): The name of the date column in the dataframe.
        norm_days (int): A number of days to normalise to (e.g. to scale results between time periods)
        from_date (str, default='2020-03-01'): The start of the date range to consider.
        to_date (str, default=now): The end of the date range to consider.
    Returns:
        activity (float): Total activity over the given time period, scaled to another time period.
    """
    _date = objs[datefield]
    total_days = (pd.to_datetime(to_date) - pd.to_datetime(from_date)).days
    norm = norm_days / total_days
    slicer = (_date > pd.to_datetime(from_date)) & (_date < pd.to_datetime(to_date))
    activity = labels[slicer].sum(axis=0).sort_values()
    return norm * activity


def thematic_diversity(objs, labels, datefield, covid_slicer, from_date='2020-03-01'):
    '''Calculate the Shannon diversity of objects by topic, for objects tagged as covid-related and non-covid-related.
    

    Args:
        objs (DataFrame): Objects over which to calculate total activity
        labels (int): CorEx's binary labels matrix, provided by CorEx.
        datefield (str): The name of the date column in the dataframe.
        covid_slicer (Series): boolean indexer, indicating projects tagged as covid and non-covid related.
        from_date (str, default='2020-03-01'): The start of the date range to consider.
    Returns:
        diversity_covid, diversity_noncovid: Thematic diversity for covid-related and non-covid-related objects.
    '''
    _date = objs[datefield]
    date_slicer = _date > pd.to_datetime(from_date)
    thematic_diversity_covid = shannon(labels.loc[date_slicer & covid_slicer].sum(axis=0))
    thematic_diversity_noncovid = shannon(labels.loc[date_slicer & ~covid_slicer].sum(axis=0))
    return thematic_diversity_covid, thematic_diversity_noncovid


def covid_filterer(topic):
    '''Rough rule for determining whether a topic is covid-related. Returns a boolean indexer.'''
    covid_terms = ['covid', 'covid-19', 'coronavirus']
    return any(term in topic for term in covid_terms)


def _generate_indicators(objects, topic_labels, datefield, old_from_date, old_to_date, weight_field, norm_days):
    """Generate a suite of indicators for a given set of objects"""
    
    # Apply filter to data based on topic labels and date
    covid_label = list(filter(covid_filterer, topic_labels))[0]  # Work out which is the covid topic
    slicer = topic_labels[covid_label] == 1  # Slice by being covid or non-covid related
    weight = 1 if weight_field is None else objects[weight_field]  # i.e. funding, if specified, instead of raw counts
    topic_labels = topic_labels.multiply(weight, axis=0)  # scale binary topic-label matrix by weights, if specified

    # Convenience methods for getting total activity
    get_total_activity_all = partial(total_activity_by_topic_in_date_range, objects, topic_labels, datefield, norm_days)
    get_total_activity_cov = partial(total_activity_by_topic_in_date_range, objects.loc[slicer], topic_labels.loc[slicer], datefield, norm_days)
    get_total_activity_noncov = partial(total_activity_by_topic_in_date_range, objects.loc[~slicer], topic_labels.loc[~slicer], datefield, norm_days)

    # Supposed to represent 2015-2019
    old_dates = dict(from_date=old_from_date, to_date=old_to_date)

    # Let's make indicators
    indicators = {}

    # Levels of activity by topic, total (2020)
    indicators["total_activity"] = get_total_activity_all()  # Uses default dates (March 2020 til present)

    # Levels of activity by topic, relative to the average from 2015-2019
    _norm_past_activity = get_total_activity_all(**old_dates)
    _norm_past_activity.loc[_norm_past_activity == 0] = 1 # In the case where there is no past activity, take 1 as an upper bound
    indicators["relative_activity"] = indicators["total_activity"] / _norm_past_activity

    # Levels of activity by topic, relative to the average from 2015-2019, covid tagged
    _total_activity = get_total_activity_cov()
    _norm_past_activity = get_total_activity_cov(**old_dates)
    _norm_past_activity.loc[_norm_past_activity == 0] = 1 # In the case where there is no past activity, take 1 as an upper bound
    indicators["relative_activity_covid"] = _total_activity / _norm_past_activity

    # Levels of activity by topic, relative to the average from 2015-2019, non-covid tagged
    _total_activity = get_total_activity_noncov()
    _norm_past_activity = get_total_activity_noncov(**old_dates)
    _norm_past_activity.loc[_norm_past_activity == 0] = 1 # In the case where there is no past activity, take 1 as an upper bound
    indicators["relative_activity_noncovid"] = _total_activity / _norm_past_activity

    # Overrepresentation (activity) of covid tagged compared to non-covid tagged projects
    denominator = indicators["relative_activity_noncovid"].copy()
    denominator.loc[denominator == 0] = 1 # In the case where there is no past activity, take 1 as an upper bound
    indicators["overrepresentation_activity"] = indicators["relative_activity_covid"] / denominator

    # Thematic diversity (activity) of co-occurring topics of covid tagged compared to non-covid tagged projects
    thematic_diversity_covid, thematic_diversity_noncovid = thematic_diversity(objects, topic_labels,
                                                                               datefield, slicer)
    indicators["thematic_diversity"] = {"covid-related-projects": thematic_diversity_covid,
                                        "non-covid-related-projects": thematic_diversity_noncovid}
    return indicators


def generate_indicators(data_getter, topic_model_path, datefield, old_from_date, old_to_date, covid_start, weight_field=None, geo_split=False):
    covid_end = dt.now()
    norm_days = (pd.to_datetime(covid_end) - pd.to_datetime(covid_start)).days
    objects, topic_labels = get_objects_and_labels(data_getter, topic_model_path)
    if geo_split:
        indicators = {geo_label: _generate_indicators(objects.loc[geo_index], topic_labels.loc[geo_index],
                                                      datefield, old_from_date, old_to_date, weight_field, norm_days)
                      for geo_index, geo_label in data_getter(geo_split=True)}
        return indicators
    else:
        return _generate_indicators(objects, topic_labels, datefield, old_from_date, old_to_date, weight_field, norm_days)


def flatten(nested_dict):
    items = []
    for k, v in nested_dict.items():
        if type(v) is pd.Series:
            v = dict(v)
        if type(v) is dict:
            for v in flatten(v):
                items.append((k, *v))
        else:
            items.append((k, v))
    return items


def make_indicator_description(indc_name, ds_name):
    return VERBOSE_INDICATOR_NAMES[indc_name].format(ENTITY_TYPE[ds_name])


def make_ctry_metadata(ctry_code):
    nuts_lookup = get_nuts_info_lookup()  # NB: lru_cached
    if ctry_code.startswith('iso_'):
        ctry_code = ctry_code[4:]
        nuts_level = 0
        nuts_name = country_iso_code_to_name(ctry_code, iso2=True)
        filename = 'by-country.csv'
    else:
        nuts_level = len(ctry_code) - 1
        nuts_name = nuts_lookup[ctry_code]['nuts_name']
        filename = f'nuts-{nuts_level}.csv'
    return ctry_code, nuts_level, nuts_name, filename


if __name__ == "__main__":

    import boto3
    from pathlib import Path

    # Generate all indicators
    all_indicators = {}
    for params in DATA_PARAMS:
        weight_field = params.pop('weight_field')
        entity_type = params.pop('entity_type')
        dataset = params.pop('dataset')
        topic_model_path=BASE_PATH.format(dataset) + "/{}"
        kwargs = dict(topic_model_path=topic_model_path, **BASE_KWARGS, **params)
        all_indicators[dataset] = generate_indicators(**kwargs)
        if weight_field is not None:
            all_indicators[f'{dataset}-funding'] = generate_indicators(weight_field=weight_field, **kwargs)


    # Prepare and format the data, with metadata
    output = defaultdict(list)  # mapping of filepath --> [meta]data
    for ds_name, ctry_code, indc_name, topic_name, indc_value in flatten(all_indicators):
        if pd.isnull(indc_value) or indc_value == 0:
            continue
        indc_desc = make_indicator_description(indc_name, ds_name)
        ctry_code, nuts_level, nuts_name, filename = make_ctry_metadata(ctry_code)
        pseudofile = output[f'{ds_name}/{topic_name}/{filename}']
        pseudofile.append({'indicator_name': indc_name,
                           'indicator_value': indc_value,
                           'indicator_description': indc_desc,
                           'nuts_level': nuts_level,
                           'nuts_code': ctry_code,
                           'nuts_name': nuts_name})

    # Save the data to disk and S3
    s3 = boto3.resource('s3')
    all_paths = []
    for path, data in output.items():
        df = pd.DataFrame(data)
        df = df.sort_values(by=["indicator_name", "nuts_level", "nuts_code"]).reset_index(drop=True).dropna(axis=0)
        if len(df) == 0:
            continue
        Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        s3.Bucket('eurito-csv-indicators').upload_file(path, path)
        all_paths.append(f'https://eurito-csv-indicators.s3-eu-west-1.amazonaws.com/{path}')
        del df
