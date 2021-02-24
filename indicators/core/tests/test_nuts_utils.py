from indicators.core.nuts_utils import NutsFinder, get_nuts_info_lookup, iso_to_nuts
from indicators.core.config import EU_COUNTRIES


def test_NutsFinder():
    assert NutsFinder() is NutsFinder()  # i.e. check the cache works


def test_get_nuts_info_lookup():
    assert (
        get_nuts_info_lookup() is get_nuts_info_lookup()
    )  # i.e. check the cache works
    lookup = get_nuts_info_lookup()
    assert lookup["UK"]["nuts_name"] == "United Kingdom"
    assert lookup["UKD7"]["nuts_name"] == "Merseyside"
    assert lookup["BA"]["nuts_name"] == "Bosnia and Herzegovina"
    assert len(lookup) > 2000


def test_iso_to_nuts():
    assert iso_to_nuts("GB") == "UK"
    assert iso_to_nuts("GR") == "EL"
    for iso in EU_COUNTRIES:
        if iso in ("GB", "GR"):
            continue
        assert iso_to_nuts(iso) == iso
