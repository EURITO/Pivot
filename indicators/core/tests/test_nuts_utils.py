from unittest import mock
from indicators.core.nuts_utils import (
    NutsFinder,
    get_nuts_info_lookup,
    iso_to_nuts,
    make_reverse_lookup,
    get_geo_lookup,
)
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


def test_make_reverse_lookup():
    a = ("a", "AA")
    b = ("b", "BB")
    data = [(1, a), (2, b), (3, a), (4, a)]
    assert make_reverse_lookup(data) == {str(a): {1, 3, 4}, str(b): {2}}

    def getter(item):
        return item[1][1]

    assert make_reverse_lookup(data, key=getter) == {
        "AA": {1, 3, 4},
        "BB": {2},
    }
    assert make_reverse_lookup(data, key=getter, prefix="something_") == {
        "something_AA": {1, 3, 4},
        "something_BB": {2},
    }


def test_get_geo_lookup():
    mocked_module = mock.MagicMock()
    mocked_module.get_iso2_to_id.return_value = [("Something else", "GB")]
    mocked_module.get_lat_lon.return_value = [
        ("58VE", 51.400, -0.1095),
        ("Potterow", 55.9462, -3.1872),
    ]
    assert get_geo_lookup(mocked_module) is get_geo_lookup(mocked_module)
    assert get_geo_lookup(mocked_module) == {
        "UK": {"58VE", "Potterow"},
        "UKI": {"58VE"},
        "UKI6": {"58VE"},
        "UKI62": {"58VE"},
        "UKM": {"Potterow"},
        "UKM7": {"Potterow"},
        "UKM75": {"Potterow"},
        "iso_GB": {"Something else"},
    }
