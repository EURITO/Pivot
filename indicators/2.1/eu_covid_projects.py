
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
from transformers.modeling_auto import AutoModelForQuestionAnswering
from transformers.modeling_bart import BartForSequenceClassification
import spacy

from nesta.core.orms.orm_utils import db_session, get_mysql_engine
from nesta.core.orms.cordis_orm import Project

requests_cache.install_cache(expire_after=timedelta(weeks=1))

CORDIS_API = 'https://cordis.europa.eu/api/search/results'
CORDIS_QUERY = ("contenttype='project' AND "
                "/project/relations/categories/nature/code='{crisis_code}'")
CRISIS_CODES = ("crisisRecovery", "crisisResponse", "crisisPreparedness")
RETRY_PROTOCOL = Retry(total=5, backoff_factor=0.1)

# %%


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

# %%


@lru_cache()
def load_bart(model_name="facebook/bart-large-mnli"):
    """[summary]

    Args:
        model_name (str, optional): [description]. Defaults to "facebook/bart-large-mnli".

    Returns:
        [type]: [description]
    """
    with requests_cache.disabled():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BartForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

    # with requests_cache.disabled():
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # return tokenizer, model


# %%


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

# %%


def assign_probability(premise, hypothesis_label):
    """[summary]

    Args:
        premise ([type]): [description]
        hypothesis_label ([type]): [description]

    Returns:
        [type]: [description]
    """
    tokenizer, model = load_bart()
    hypothesis = f'This text is about {hypothesis_label}.'

    # run through model pre-trained on MNLI
    # input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
    # logits = model(input_ids)[0]
    to_encode = [(p, hypothesis) for p in premise]

    tokens = tokenizer.batch_encode_plus(
        to_encode,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_token_type_ids=True,
        return_tensors="pt")
    input_ids = tokens['input_ids']
    logits = model(input_ids)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=-1)
    true_probs = [p.item() for p in probs[:, 1]]
    return true_probs

    # entail_contradiction_logits = logits[:, [0, 2]]
    # probs = entail_contradiction_logits.softmax(dim=1)
    # true_prob = probs[:, 1].item()
    # return true_prob


def evaluate_probability(text, hypothesis_label, percentile=90):
    """[summary]

        Returns:
            [type]: [description]
    """
    # probs = [assign_probability(str(sent), hypothesis_label)
    #         for sent in nlp(text).sents]
    premise = [str(sent) for sent in nlp(text).sents]
    probs = assign_probability(premise, hypothesis_label)
    return np.percentile(probs, q=percentile)

# %%


@ lru_cache()
def get_cordis_projects():
    """[summary]

    Returns:
        [type]: [description]
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(Project.rcn, Project.objective,
                              Project.title, Project.start_date_code)
        return [dict(rcn=rcn, text=text, title=title, start_date=start_date)
                for rcn, text, title, start_date in query.all()]

# %%


def label_cordis_projects(test=False, test_limit=10):
    """[summary]

    Args:
        test (bool, optional): [description]. Defaults to False.
        test_limit (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    projects = get_cordis_projects()
    for i, proj in enumerate(projects):
        for crisis_code in CRISIS_CODES:
            crisis_code = camel_to_snake(crisis_code)
            label = f'covid {crisis_code}'
            prob = evaluate_probability(proj['text'], label)
            proj[crisis_code] = prob
        if test and i == test_limit:
            projects = projects[:test_limit]
            break
    return projects

# %%

# %%
