"""
nuts_utils
==========

Tools for dealing with NUTS regions.
"""

from functools import lru_cache
from indicators.core.config import EU_COUNTRIES
from indicators.core.config import NUTS_EDGE_CASES

from nuts_finder import NutsFinder as _NutsFinder


@lru_cache()
def NutsFinder():
    """Retrieve a NutsFinder instance and cache it.

    Returns:
        A NutsFinder instance
    """
    return _NutsFinder()


@lru_cache()
def get_nuts_info_lookup():
    """Generate a lookup table of nuts ID to nuts info (name, level, code)"""
    nf = NutsFinder()
    lookup = {
        item["properties"]["NUTS_ID"]: {
            "nuts_name": item["properties"]["NAME_LATN"],
            "nuts_level": item["properties"]["LEVL_CODE"],
            "nuts_code": item["properties"]["NUTS_ID"],
        }
        for item in nf.shapes["features"]
    }
    # Add edge-cases
    lookup.update(**NUTS_EDGE_CASES)
    return lookup


def iso_to_nuts(iso_code):
    """Convert an ISO2 code into a NUTS code
    (actually only does this for GB and GR, and
    returns None for non-EU countries)

    Args:
        iso_code (str): ISO2 code

    Returns:
        nuts_code
    """
    # Convert to NUTS code if in EU
    if iso_code not in EU_COUNTRIES:
        return None
    nuts_code = iso_code  # default
    # GB and GR iso and nuts1 codes don't match
    if iso_code == "GB":
        nuts_code = "UK"
    elif iso_code == "GR":
        nuts_code = "EL"
    return nuts_code
