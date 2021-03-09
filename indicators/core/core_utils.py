from indicators.core.nuts_utils import get_geo_lookup
from pathlib import Path
import pandas as pd


def object_getter(topic_module, from_date="2015-01-01", geo_split=False):
    """Get all object from a given start date. `geo_split`
    alters the behaviour of the function, such that `geo_split=False`
    will yield an article, whereas `geo_split=False` yields
    a tuple of (indexer, geo_code) where indexer can be used to slice
    `articles`

    Args:
        from_date (str, optional): Min object creation date. Defaults to "2015-01-0\
1".
        geo_split (bool, optional): Alter the behaviour to return an indexer
                                    by geography code. Defaults to False.

    Yields:
        objects (list(dict))
    """
    objects = topic_module.get_objects(from_date=from_date)
    if geo_split:
        nuts_to_id_lookup = get_geo_lookup(topic_module)
        for geo_code, ids in nuts_to_id_lookup.items():
            indexes = list(article["id"] in ids for article in objects)
            yield indexes, geo_code
    else:
        yield objects


def flatten(nested_dict):
    """Convert nested dictionary into flat list of tuples.
    E.g.
        {'a': {'b': 'c', 'd': 'e'}}

    becomes
        [('a', 'b', 'c'), ('a', 'd', 'e')]
    """
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
