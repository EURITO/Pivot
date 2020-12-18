
# %%
'''
EU Covid Projects
=================

Code to get Cordis projects and label them as being "Covid" and "non-Covid" according
to Cordis's hard labels.
'''

from datetime import timedelta
from functools import lru_cache
import re
import numpy as np
import requests_cache
from requests.adapters import HTTPAdapter, Retry
from transformers import AutoTokenizer
from transformers.modeling_bart import BartForSequenceClassification

from indicators.core.nlp_utils import fit_topics, vectorise_docs

from nesta.core.orms.orm_utils import db_session, get_mysql_engine
from nesta.core.orms.cordis_orm import Project

requests_cache.install_cache(expire_after=timedelta(weeks=1))

CORDIS_API = 'https://cordis.europa.eu/api/search/results'
CORDIS_QUERY = ("contenttype='project' AND "
                "/project/relations/categories/nature/code='{crisis_code}'")
CRISIS_CODES = ("crisisRecovery", "crisisResponse", "crisisPreparedness")
RETRY_PROTOCOL = Retry(total=5, backoff_factor=0.1)


@lru_cache()
def camel_to_snake(name):
    """[summary]

    Args:
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', name).lower()


def persistent_get(*args, **kwargs):
    """Use a decent retry protocol to persistently make get requests"""
    session = requests_cache.CachedSession()
    session.mount('https://', HTTPAdapter(max_retries=RETRY_PROTOCOL))
    response = session.get(*args, **kwargs)
    response.raise_for_status()
    return response


def hit_api(crisis_code, page_num=1, page_size=100):
    """Format and execute the API request"""
    query = CORDIS_QUERY.format(crisis_code=crisis_code)
    params = {"q": query, "p": page_num, "num": page_size}
    response = persistent_get(CORDIS_API, params=params)
    data = response.json()
    return data["payload"]


def get_projects_by_code(crisis_code, page_size=100):
    """[summary]

    Args:
        crisis_code ([type]): [description]
        page_size (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    data, page_num = [], 0
    while "reading_data":
        _data = hit_api(crisis_code=crisis_code, page_num=page_num,
                        page_size=page_size)
        results = _data["results"]
        # Unpack data
        data += results
        if len(results) < page_size:
            break
        page_num += 1
    return data


def get_rcns_by_code(crisis_code, page_size=100):
    """[summary]

    Args:
        crisis_code ([type]): [description]
        page_size (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    return [project['rcn']
            for project in get_projects_by_code(crisis_code, page_size=page_size)]


@lru_cache()
def get_crisis_rcns(page_size=100):
    """[summary]

    Args:
        page_size (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    data = {crisis_code: get_rcns_by_code(crisis_code=crisis_code, page_size=page_size)
            for crisis_code in CRISIS_CODES}
    return data


@ lru_cache()
def get_cordis_projects(from_date="2015-01-01"):
    """[summary]

    Returns:
        [type]: [description]
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(Project.rcn, Project.objective,
                              Project.title, Project.start_date_code,
                              Project.total_cost)
        query = query.filter(Project.start_date_code > from_date)
        return [dict(rcn=rcn, text=text, title=title,
                     start_date=start_date, funding=funding)
                for rcn, text, title, start_date, funding in query.all()]


def fit_cordis_topics(n_topics=150):
    """[summary]

    Args:
        n_topics (int, optional): [description]. Defaults to 150.

    Returns:
        [type]: [description]
    """
    projects = get_cordis_projects()
    doc_vectors, feature_names = vectorise_docs([p['text'] for p in projects])
    titles = [p['title'] for p in projects]
    anchors = [['covid', 'covid_19', "coronavirus", '2019_ncov', 'sars_cov_2', 'infection', 'immunity'],
               ['cells', 'cell', 'cellular', 'proteins', 'in_vivo',
                'in_vitro', 'protein', 'mouse', 'expression']]
    topic_model = fit_topics(dataset_label="cordis", doc_vectors=doc_vectors,
                             feature_names=feature_names, titles=titles,
                             n_topics=n_topics, anchors=anchors, anchor_strength=50)
    return projects, topic_model
