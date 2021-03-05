from unittest import mock
from indicators.two.cordis_topics import (
    get_nuts_to_id,
    get_iso2_to_id,
    get_objects,
)

PATH = "indicators.two.cordis_topics.{}"


@mock.patch(PATH.format("get_iso2_to_id"))
def test_get_nuts_to_id(mocked_get_iso2_to_id):
    mocked_get_iso2_to_id.return_value = [
        ("first", "GB"),
        ("second", "FR"),
        ("third", "IT"),
    ]
    assert get_nuts_to_id() == [
        ("first", "UK"),  # <-- note change
        ("second", "FR"),
        ("third", "IT"),
    ]


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_iso2_to_id(mocked_db_session, mocked_get_mysql_engine):
    query = mocked_db_session().__enter__().query().join().filter()
    query.all.return_value = ["BREADCRUMB"]
    assert get_iso2_to_id() is get_iso2_to_id()
    assert get_iso2_to_id() == ["BREADCRUMB"]


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_objects(mocked_db_session, mocked_get_mysql_engine):
    query = mocked_db_session().__enter__().query().filter()
    query.all.return_value = [
        (1, "the text", "the title", "01-01-2020", "funding1"),
        (2, "more text", "more title", "02-01-2020", "funding2"),
    ]
    assert get_objects("01-01-2020") is get_objects("01-01-2020")  # ie cache is working
    assert get_objects("01-01-2020") == [
        {
            "rcn": 1,
            "text": "the text",
            "title": "the title",
            "created": "01-01-2020",
            "funding": "funding1",
        },
        {
            "rcn": 2,
            "text": "more text",
            "title": "more title",
            "created": "02-01-2020",
            "funding": "funding2",
        },
    ]
