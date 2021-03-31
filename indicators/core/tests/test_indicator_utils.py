from unittest import mock

from indicators.two import arxiv_topics, nih_topics, cordis_topics
from indicators.core.indicator_utils import (
    make_indicator_description,
    make_ctry_metadata,
    prepare_file_data,
    sort_and_filter_data,
    _days_of_covid,
    save_and_upload,
    safe_divide,
    sort_save_and_upload,
    pd,
    INDICATORS,
)

PATH = "indicators.core.indicator_utils.{}"


def test_make_indicator_description():
    for module in (arxiv_topics, nih_topics, cordis_topics):
        label = module.model_config["metadata"]["entity_type"]
        description = make_indicator_description("total_activity", label)
        assert description[0:16] == "Total activity ("
        assert description[-18:] == ") since March 2020"
        assert len(description) > 40 and len(description) < 50


def test_make_ctry_metadata():
    assert make_ctry_metadata("FR") == {
        "nuts_code": "FR",
        "nuts_level": 1,
        "nuts_name": "France",
        "filename": "nuts-1.csv",
    }
    assert make_ctry_metadata("FR1") == {
        "nuts_code": "FR1",
        "nuts_level": 2,
        "nuts_name": "Ile-de-France",
        "filename": "nuts-2.csv",
    }
    assert make_ctry_metadata("FR10") == {
        "nuts_code": "FR10",
        "nuts_level": 3,
        "nuts_name": "Ile-de-France",
        "filename": "nuts-3.csv",
    }
    assert make_ctry_metadata("FR101") == {
        "nuts_code": "FR101",
        "nuts_level": 4,
        "nuts_name": "Paris",
        "filename": "nuts-4.csv",
    }
    assert make_ctry_metadata("iso_FR") == {
        "nuts_code": "FR",
        "nuts_level": 0,
        "nuts_name": "France",
        "filename": "by-country.csv",
    }


@mock.patch(PATH.format("make_indicator_description"))
@mock.patch(PATH.format("make_ctry_metadata"))
def test_prepare_file_data(mocked_metadata, mocked_description):
    mocked_description.return_value = "DESCRIPTION"
    mocked_metadata.side_effect = lambda x: {
        "nuts_code": "FR",
        "nuts_level": 0,
        "nuts_name": "France",
        "filename": "by-country.csv",
    }
    indicators = {
        "my_data_set": {
            "an entity type": {
                "a country": {
                    "an indicator": {"a topic": 123},
                    "another indicator": {"a topic": None},
                },
                "another country": {"another indicator": {"another topic": 234}},
            },
        },
        "their_data_set": {
            "another entity type": {
                "a country": {
                    "an indicator": {"a topic": 0},
                    "another indicator": {"another topic": 123},
                },
                "another country": {"another indicator": {"a topic": None}},
            },
        },
    }
    assert prepare_file_data(indicators) == {
        "my_data_set/a topic/by-country.csv": [
            {
                "indicator_name": "an indicator",
                "indicator_value": 123,
                "indicator_description": "DESCRIPTION",
                "nuts_code": "FR",
                "nuts_level": 0,
                "nuts_name": "France",
            }
        ],
        "my_data_set/another topic/by-country.csv": [
            {
                "indicator_name": "another indicator",
                "indicator_value": 234,
                "indicator_description": "DESCRIPTION",
                "nuts_code": "FR",
                "nuts_level": 0,
                "nuts_name": "France",
            }
        ],
        "their_data_set/another topic/by-country.csv": [
            {
                "indicator_name": "another indicator",
                "indicator_value": 123,
                "indicator_description": "DESCRIPTION",
                "nuts_code": "FR",
                "nuts_level": 0,
                "nuts_name": "France",
            }
        ],
    }


def test_sort_and_filter_data():
    file_data = {
        "path1": [
            {
                "indicator_name": "first",
                "nuts_level": 1,
                "nuts_code": "FR",
                "indicator_value": 10,
            }
        ],
        "path2": [
            {
                "indicator_name": "second",
                "nuts_level": 2,
                "nuts_code": "GB",
                "indicator_value": 0.1203,
            }
        ],
    }
    sorted_data = sort_and_filter_data(file_data)
    assert sorted_data.keys() == file_data.keys()
    assert list(sorted_data["path1"].columns) == list(sorted_data["path2"].columns)
    assert sorted(list(sorted_data["path1"].columns)) == sorted(
        [
            "indicator_name",
            "nuts_level",
            "nuts_code",
            "indicator_value",
        ]
    )


@mock.patch(PATH.format("boto3"))
@mock.patch(PATH.format("Path"))
def test_save_and_upload(mocked_path, mocked_boto):
    sorted_file_data = {"path1": mock.Mock(), "path2": mock.Mock()}
    bucket = mocked_boto.resource().Bucket()
    save_and_upload(sorted_file_data)
    for path, data in sorted_file_data.items():
        assert data.to_csv.call_count == 1
    assert bucket.upload_file.call_count == 2


@mock.patch(PATH.format("INDICATORS"))
def test_days_of_covid(mocked_INDICATORS):
    mocked_INDICATORS.__getitem__.return_value = {
        "from_date": "01-01-1900",
        "to_date": "28-01-1900",
    }
    assert _days_of_covid() == 28

    mocked_INDICATORS.__getitem__.return_value = {
        "from_date": "01-01-1901",
        "to_date": "31-12-1901",
    }
    assert _days_of_covid() == 365


def test_safe_divide():
    num = pd.Series({"a": 2, "b": 3, "c": 0, "d": 4})
    den = pd.Series({"a": 4, "b": 0, "c": 0, "d": 10})
    exp = pd.Series({"a": 0.5, "b": 3, "c": 0, "d": 0.4})
    res = safe_divide(num, den)
    assert (exp == res).all()


@mock.patch(PATH.format("prepare_file_data"))
@mock.patch(PATH.format("sort_and_filter_data"))
@mock.patch(PATH.format("save_and_upload"))
def test_sort_save_and_upload(mocked_save, mocked_sort, mocked_prepare):
    mocked_save.side_effect = lambda x: x
    mocked_sort.side_effect = lambda x: x
    mocked_prepare.side_effect = lambda x: x
    sort_save_and_upload("dummy")
    assert mocked_save.call_count == mocked_sort.call_count == mocked_prepare.call_count
    assert mocked_save.call_args == mocked_sort.call_args == mocked_prepare.call_args
