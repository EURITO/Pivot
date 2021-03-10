"""
nuts_utils
==========

Tools for dealing with NUTS regions.
"""

from functools import lru_cache
from indicators.core.config import EU_COUNTRIES
from indicators.core.config import NUTS_EDGE_CASES
from itertools import groupby
from operator import itemgetter
from nuts_finder import NutsFinder as _NutsFinder
import logging


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
    logging.info("Generating global NUTS lookup")
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


def make_reverse_lookup(data, key=itemgetter(1), prefix=""):
    """Group an iterable of the form (`id`, `code_object`),
    where a `code` exists somewhere in `code_object` (extracted
    with `key`) such that the output is of the form:

       {code: {id1, id2, id3}}

    You might think of this as being {nuts_code: {article ids}}

    Args:
      data: Iterable of the form (`id`, `code_object`)
      key: Function for extracting `code` from `code_object`
      prefix: Prefix for the code, if required (intended to distinguish
              NUTS from ISO codes) (Default value = "")

    Returns:
       Grouped object of the form {code: {id1, id2, id3}}
    """
    # Look nuts code --> article id
    logging.info("Performing reverse lookup")    
    data_wout_null_codes = filter(key, data)  # without null codes
    return {
        f"{prefix}{code}": set(id for id, _ in group)
        for code, group in groupby(sorted(data_wout_null_codes, key=key), key=key)
    }


@lru_cache()
def get_geo_lookup(module):
    """Generate a geographic lookup for a given topic_module (e.g. arxiv_topics)
    which has a get_lat_lon and get_iso2_to_id method. Output of the form:

        {geography_code: [object_id]}

    where the object_id could be e.g. an article ID in the case of arXiv,
    or a project ID in the case of NiH or Cordis. NUTS codes are given as is,
    and ISO2 codes are prefixed with "iso_" to distinguish them from NUTS
    regions (avoiding any unforeseen ISO-NUTS code clashes).
    """
    # Forward lookup
    nf = NutsFinder()
    id_to_nuts_lookup = {
        id: nf.find(lat=lat, lon=lon) for id, lat, lon in module.get_lat_lon()
    }
    id_nuts = [  # splatten out the nuts IDs, ready for grouping
        (id, info["NUTS_ID"])
        for id, nuts_info in id_to_nuts_lookup.items()
        for info in nuts_info
    ]
    id_iso2 = module.get_iso2_to_id()
    # Reverse lookups
    nuts_to_id_lookup = make_reverse_lookup(id_nuts)
    iso2_to_id_lookup = make_reverse_lookup(id_iso2, prefix="iso_")
    # Combine lookups and return
    return {**nuts_to_id_lookup, **iso2_to_id_lookup}
