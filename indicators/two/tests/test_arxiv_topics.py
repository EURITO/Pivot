from unittest import mock
from indicators.two.arxiv_topics import get_lat_lon, get_iso2_to_id, get_objects

PATH = "indicators.two.arxiv_topics.{}"


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_lat_lon(mocked_db_session, mocked_get_mysql_engine):
    query = mocked_db_session().__enter__().query()
    # 2 joins, 5 filters, 1 groupby, 1 all
    joined = query.join().join()
    filtered = joined.filter().filter().filter().filter().filter()
    grouped = filtered.group_by()
    grouped.all.return_value = ["BREADCRUMB"]
    # Result will be False unless the above pattern has been fulfilled
    assert get_lat_lon() == ["BREADCRUMB"]


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_iso2_to_id(mocked_db_session, mocked_engine):
    query = mocked_db_session().__enter__().query().join()
    query.all.return_value = ["BREADCRUMB"]
    assert get_iso2_to_id() == ["BREADCRUMB"]


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_objects(mocked_db_session, mocked_get_mysql_engine):
    query = mocked_db_session().__enter__().query().filter().filter()
    query.all.return_value = [
        (1, "some text", "a title", "01-01-2020"),
        (2, "more text", "another title", "02-01-2020"),
    ]
    assert get_objects("01-01-2020") is get_objects("01-01-2020")  # ie cache is working
    assert get_objects("01-01-2020") == [
        {"id": 1, "text": "some text", "title": "a title", "created": "01-01-2020"},
        {
            "id": 2,
            "text": "more text",
            "title": "another title",
            "created": "02-01-2020",
        },
    ]
