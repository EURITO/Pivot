"""
"""

from functools import lru_cache
from indicators.core.constants import EU_COUNTRIES

from nuts_finder import NutsFinder as _NutsFinder


@lru_cache
def NutsFinder():
    """[summary]

    Returns:
        [type]: [description]
    """
    return _NutsFinder()


@lru_cache
def get_nuts_info_lookup():
    """[summary]

    Returns:
        [type]: [description]
    """
    nf = NutsFinder()
    lookup = {item['properties']['NUTS_ID']: {'nuts_name': item['properties']['NAME_LATN'],
                                              'nuts_level': item['properties']['LEVL_CODE'],
                                              'nuts_code': item['properties']['NUTS_ID']}
              for item in nf.shapes['features']}
    return lookup


def iso_to_nuts(iso_code):
    """[summary]

    Args:
        iso_code (bool): [description]

    Returns:
        [type]: [description]
    """
    # Convert to NUTS code if in EU
    if iso_code not in EU_COUNTRIES:
        return None
    nuts_code = iso_code
    # GB and GR iso and nuts1 codes don't match
    if iso_code == 'GB':
        nuts_code = 'UK'
    elif iso_code == 'GR':
        nuts_code = 'EL'
    return nuts_code
