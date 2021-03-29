from indicators.core.config import INDICATORS
from indicators.core.nuts_utils import get_geo_lookup


def object_getter(topic_module, geo_split=False):
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
    from_date = INDICATORS["precovid_dates"]["from_date"]
    objects = topic_module.get_objects(from_date=from_date)
    if geo_split:
        nuts_to_id_lookup = get_geo_lookup(topic_module)
        for geo_code, ids in nuts_to_id_lookup.items():
            indexes = list(article["id"] in ids for article in objects)
            yield indexes, geo_code
    else:
        yield objects
