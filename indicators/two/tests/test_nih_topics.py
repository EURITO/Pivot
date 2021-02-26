from unittest import mock
from indicators.two.nih_topics import (
    get_projects,
    get_lat_lon,
    get_iso2_to_id,
    get_objects,
)

PATH = "indicators.two.nih_topics.{}"


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_projects(mocked_db_session, mocked_get_mysql_engine):
    query = mocked_db_session().__enter__().query()
    query.all.return_value = ["BREADCRUMB"]
    # Result will be False unless the above pattern has been fulfilled
    assert get_projects == ["BREADCRUMB"]


@mock.patch(PATH.format("get_projects"))
def test_get_lat_lon(mocked_get_projects):
    data = [
        (1, {"lat": "100.44", "lon": "200.24"}, True, "A"),
        (2, {"lat": "200.44", "lon": "400.24"}, False, "B"),
        (3, None, True, "C"),
        (4, {"lat": "400.44", "lon": "800.24"}, True, "D"),
        (5, None, False, "E"),
    ]
    mocked_get_projects.return_value = data
    assert get_lat_lon() == [(1, 100.44, 200.24), (4, 400.44, 800.24)]


@mock.patch(PATH.format("get_projects"))
def test_get_iso2_to_id(mocked_get_projects):
    data = [
        (1, {"lat": "100.44", "lon": "200.24"}, True, "A"),
        (2, {"lat": "200.44", "lon": "400.24"}, False, "B"),
        (3, None, True, "C"),
        (4, {"lat": "400.44", "lon": "800.24"}, True, "D"),
        (5, None, False, "E"),
    ]
    mocked_get_projects.return_value = data
    assert get_iso2_to_id() == [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_objects(mocked_db_session, mocked_get_mysql_engine):
    query = mocked_db_session().__enter__().query().filter()
    query.all.return_value = [
        (1, "phr text", "abstract text", "title text", "01-01-2020", "funding1"),
        (
            2,
            "more phr text",
            "more abstract text",
            "more title text",
            "02-01-2020",
            "funding2",
        ),
    ]
    assert get_objects("01-01-2020") is get_objects("01-01-2020")  # ie cache is working
    assert get_objects("01-01-2020") == [
        {
            "id": 1,
            "text": "phr text abstract text",
            "title": "title text",
            "created": "01-01-2020",
            "funding": "funding1",
        },
        {
            "id": 2,
            "text": "more phr text more abstract text",
            "title": "more title text",
            "created": "02-01-2020",
            "funding": "funding2",
        },
    ]
