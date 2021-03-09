import pytest
from unittest import mock
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal
from indicators.two.thematic_indicators import (
    sum_activity,
    pd,
    covid_filterer,
    covid_topic_indexer,
    relative_activity,
    get_objects_and_topics,
    generate_indicators,
    thematic_diversity,
    indicators_by_geo,
    make_indicators,
)
from indicators.two import arxiv_topics, nih_topics, cordis_topics

PATH = "indicators.two.thematic_indicators.{}"


@pytest.fixture
def objects():
    # US formatted dates, just for this test,
    # since it uses pd.to_datetime
    data = [
        {"created": "02-01-2018", "weight": 1.2},  # precovid
        {"created": "02-01-2019", "weight": 2.3},  # precovid
        {"created": "04-01-2020", "weight": 123},  # covid
        {"created": "08-01-2020", "weight": 10},  # covid
        {"created": "02-08-2021", "weight": 1},  # covid
    ]
    df = pd.DataFrame(data)
    df.created = pd.to_datetime(df.created)
    return df


@pytest.fixture
def topic_counts():
    data = [
        {"covid": 1, "something else": 0},
        {"covid": 0, "something else": 1},
        {"covid": 0, "something else": 0},
        {"covid": 1, "something else": 1},
        {"covid": 1, "something else": 0},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def geo_index():
    data = [True, True, True, False, True]
    return pd.Series(data)


def test_sum_activity(objects, topic_counts):
    covid = sum_activity(objects, topic_counts, "covid_dates").to_dict()
    precovid = sum_activity(objects, topic_counts, "precovid_dates").to_dict()
    assert covid == {"something else": 1, "covid": 2}
    assert covid.keys() == precovid.keys()

    # Weighted values should be around 0.26 at time of writing, but will change
    # if we decide to change the config slightly
    assert all(v > 0.15 and v < 0.35 for v in precovid.values())
    # Check all values are definitely decimals (i.e. have been reweighted)
    assert all(int(v) != float(v) for v in precovid.values())


def test_covid_filterer():
    assert covid_filterer(["covid", "something else", "another"])
    assert covid_filterer(["covid-19", "something else", "another"])
    assert covid_filterer(["coronavirus", "something else", "another"])
    assert not covid_filterer(["ovid", "something else", "another"])
    assert not covid_filterer(["ovid", "conoravir", "another"])
    assert not covid_filterer(["COvid19", "another"])


def test_covid_topic_indexer(topic_counts):
    covid_indexer = covid_topic_indexer(topic_counts)
    expected = [True, False, False, True, True]
    assert (covid_indexer == expected).all()


@mock.patch(PATH.format("safe_divide"), side_effect=lambda x, y: x / y)
def test_relative_activity(mocked_divide):
    # Divide num chars in "covid_dates" by num chars in "precovid_dates"
    # i.e. 11 / 14
    relative_activity(lambda x: len(x)) == 11 / 14


@mock.patch(PATH.format("parse_clean_topics"))
def test_get_objects_and_topics(mocked_parser, objects, topic_counts, geo_index):
    topic_module = mock.Mock()
    topic_module.get_objects.return_value = [objects]
    mocked_parser.return_value = topic_counts

    # No reweight
    _objects, _topics = get_objects_and_topics(
        topic_module, geo_index, weight_field=None
    )
    assert_frame_equal(_objects, objects.loc[geo_index])
    assert_frame_equal(_topics, topic_counts.loc[geo_index])

    # With reweight
    _objects, _topics = get_objects_and_topics(
        topic_module, geo_index, weight_field="weight"
    )
    expected_counts = pd.DataFrame(  # See 'weight' field of 'objects' fixture
        [
            {"covid": 1.2, "something else": 0.0},
            {"covid": 0.0, "something else": 2.3},
            {"covid": 0.0, "something else": 0.0},
            # {"covid": 10, "something else": 10},  # Indexed out
            {"covid": 1.0, "something else": 0.0},
        ]
    )
    assert_frame_equal(_objects, objects.loc[geo_index])
    assert_frame_equal(_topics.reset_index(drop=True), expected_counts)


@mock.patch(PATH.format("thematic_diversity"))
@mock.patch(PATH.format("get_objects_and_topics"))
def test_generate_indicators(mocked_getter, mocked_diversity, objects, topic_counts):
    mocked_diversity.return_value = 123
    mocked_getter.return_value = (objects, topic_counts)
    indicators = generate_indicators(
        topic_module="dummy", geo_index="dummy", weight_field="dummy"
    )
    assert indicators.pop("thematic_diversity") == {
        "covid-related-projects": 123,
        "non-covid-related-projects": 123,
    }

    expected = {
        "total_activity": {"something else": 1.0, "covid": 2.0},
        "relative_activity": {
            "covid": 7.5,
            "something else": 3.7,
        },
        "relative_activity_covid": {"something else": 1.0, "covid": 7.5},
        "relative_activity_noncovid": {"covid": 0.0, "something else": 0.0},
        "overrepresentation_activity": {
            "covid": 7.5,
            "something else": 1.0,
        },
    }
    assert expected.keys() == indicators.keys()
    for name, _indicators in expected.items():
        assert _indicators.keys() == indicators[name].keys()
        for topic, value in _indicators.items():
            assert_almost_equal(indicators[name][topic], value, decimal=1)


def test_thematic_diversity(objects, topic_counts):
    diversity = thematic_diversity(
        objects, topic_counts, [True, True, True, True, True]
    )
    assert_almost_equal(diversity, 0.92, decimal=2)


@mock.patch(PATH.format("generate_indicators"), return_value=101)
def test_indicators_by_geo(mocked_generate):
    topic_module = mock.Mock()
    topic_module.get_objects.return_value = {
        "geo1": "geo one",
        "geo2": "geo two",
    }.items()
    assert indicators_by_geo(topic_module) == {"geo one": 101, "geo two": 101}


@mock.patch(PATH.format("indicators_by_geo"), return_value=102)
def test_make_indicators(mocked_by_geo):
    output = make_indicators(arxiv_topics, nih_topics, cordis_topics)
    assert output == {"arxiv": 102, "nih": 102, "cordis": 102}
