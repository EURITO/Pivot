
# %%
'''
EU Covid Projects
=================

Code to get Cordis projects and label them as being "Covid" and "non-Covid" according
to Cordis's hard labels.
'''

from datetime import timedelta
from functools import lru_cache
import requests_cache
from requests.adapters import HTTPAdapter, Retry
from transformers import AutoTokenizer
from transformers.modeling_bart import BartForSequenceClassification

requests_cache.install_cache(expire_after=timedelta(weeks=1))

CORDIS_API = 'https://cordis.europa.eu/api/search/results'
CORDIS_QUERY = ("contenttype='project' AND "
                "/project/relations/categories/nature/code='{crisis_code}'")
CRISIS_CODES = ("crisisRecovery", "crisisResponse", "crisisPreparedness")
RETRY_PROTOCOL = Retry(total=5, backoff_factor=0.1)

# %%


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
    return [project['rcn']
            for project in get_projects_by_code(crisis_code, page_size=page_size)]


@lru_cache()
def get_crisis_rcns(page_size=100):
    data = {crisis_code: get_rcns_by_code(crisis_code=crisis_code, page_size=page_size)
            for crisis_code in CRISIS_CODES}
    return data

# %%


@lru_cache()
def load_bart():
    with requests_cache.disabled():
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        model = BartForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli")
    return tokenizer, model

# %%


def assign_probability(premise, hypothesis_label):
    tokenizer, model = load_bart()
    hypothesis = f'This text is about {hypothesis_label}.'

    # run through model pre-trained on MNLI
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
    logits = model(input_ids)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_prob = probs[:, 1].item()
    return true_prob

# %%
