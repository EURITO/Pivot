from unittest import mock
from indicators.core.core_utils import object_getter


def test_object_getter():
    output = ["1", "two", "THREE"]
    mocked_module = mock.MagicMock()
    mocked_module.get_objects.return_value = output
    assert next(object_getter(mocked_module)) == output


@mock.patch("indicators.core.core_utils.get_geo_lookup")
def test_object_getter_split(mocked_lookup):
    mocked_module = mock.MagicMock()
    mocked_module.get_objects.return_value = [
        {"id": "1"},
        {"id": "two"},
        {"id": "THREE"},
    ]
    mocked_lookup.return_value = {"FR": {"1", "THREE"}, "DE": {"two", "1"}}
    assert list(object_getter(mocked_module, geo_split=True)) == [
        ([True, False, True], "FR"),
        ([True, True, False], "DE"),
    ]


def test_parse_corex_paths():
    pass
