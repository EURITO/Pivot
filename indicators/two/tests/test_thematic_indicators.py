import pytest
from pandas.testing import assert_series_equal
from indicators.two.thematic_indicators import sum_activity, pd


@pytest.fixture
def objects():
    # US formatted dates, just for this test,
    # since it uses pd.to_datetime
    data = [
        {"created": "02-01-2018"},  # precovid
        {"created": "02-01-2019"},  # precovid
        {"created": "04-01-2020"},  # covid
        {"created": "08-01-2020"},  # covid
        {"created": "02-08-2021"},  # covid
    ]
    df = pd.DataFrame(data)
    df.created = pd.to_datetime(df.created)
    return df


@pytest.fixture
def labels():
    data = [
        {"covid": 1, "something else": 0},
        {"covid": 0, "something else": 1},
        {"covid": 0, "something else": 0},
        {"covid": 1, "something else": 1},
        {"covid": 1, "something else": 0},
    ]
    return pd.DataFrame(data)


def test_sum_activity(objects, labels):
    covid = sum_activity(objects, labels, "covid_dates")
    precovid = sum_activity(objects, labels, "precovid_dates")
    assert_series_equal(
        covid,
        pd.Series({"something else": 1, "covid": 2}),
        check_dtype=False,
        check_exact=True,  # expecting to be round numbers
    )
    assert_series_equal(
        precovid,
        pd.Series({"covid": 0.3, "something else": 0.3}),
        check_dtype=False,
        check_exact=False,  # expecting decimals
        atol=0.1,  # i.e. +/- 0.1
    )
