from unittest import mock
from indicators.two.arxiv_topics import (
    get_arxiv_eu_insts,
    get_iso2_to_id,
    _get_arxiv_articles,
    get_arxiv_articles,
)

PATH = "indicators.two.arxiv_topics.{}"


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_arxiv_eu_insts(mocked_db_session, mocked_get_mysql_engine):
    query = mocked_db_session().__enter__().query()
    # 2 joins, 5 filters, 1 groupby, 1 all
    joined = query.join().join()
    filtered = joined.filter().filter().filter().filter().filter()
    grouped = filtered.group_by()
    grouped.all.return_value = ["BREADCRUMB"]
    # Result will be False unless the above pattern has been fulfilled
    assert get_arxiv_eu_insts() == ["BREADCRUMB"]


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_iso2_to_id(mocked_db_session, mocked_engine):
    query = mocked_db_session().__enter__().query().join()
    query.all.return_value = ["BREADCRUMB"]
    assert get_iso2_to_id() == ["BREADCRUMB"]


@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test__get_arxiv_articles(mocked_db_session, mocked_get_mysql_engine):
    query = mocked_db_session().__enter__().query().filter().filter()
    query.all.return_value = [
        (1, "some text", "a title", "01-01-2020"),
        (2, "more text", "another title", "02-01-2020"),
    ]
    assert _get_arxiv_articles("01-01-2020") is _get_arxiv_articles(
        "01-01-2020"
    )  # ie cache is working
    assert _get_arxiv_articles("01-01-2020") == [
        {"id": 1, "text": "some text", "title": "a title", "created": "01-01-2020"},
        {
            "id": 2,
            "text": "more text",
            "title": "another title",
            "created": "02-01-2020",
        },
    ]


# @mock.patch(PATH.format("_get_arxiv_articles"))
# def test_get_arxiv_articles(mocked_articles):
#     output = ["1", "two", "THREE"]
#     mocked_articles.return_value = output
#     assert next(get_arxiv_articles()) == output


# @mock.patch(PATH.format("_get_arxiv_articles"))
# @mock.patch(PATH.format("get_arxiv_geo_lookup"))
# def test_get_arxiv_articles_geo_split(mocked_lookup, mocked_articles):
#     mocked_articles.return_value = [{"id": "1"}, {"id": "two"}, {"id": "THREE"}]
#     mocked_lookup.return_value = {"FR": {"1", "THREE"}, "DE": {"two", "1"}}
#     assert list(get_arxiv_articles(geo_split=True)) == [
#         ([True, False, True], "FR"),
#         ([True, True, False], "DE"),
#     ]
