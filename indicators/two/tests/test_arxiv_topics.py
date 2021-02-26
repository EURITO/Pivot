from unittest import mock
from indicators.two.arxiv_topics import (
    get_arxiv_eu_insts,
    make_reverse_lookup,
    get_iso2_to_id_lookup,
    get_arxiv_geo_lookup,
    _get_arxiv_articles,
    get_arxiv_articles,
    fit_arxiv_topics,
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


@mock.patch(PATH.format("make_reverse_lookup"))
@mock.patch(PATH.format("get_mysql_engine"))
@mock.patch(PATH.format("db_session"))
def test_get_iso2_to_id_lookup(
    mocked_db_session, mocked_get_mysql_engine, mocked_reverse_lookup
):
    query = mocked_db_session().__enter__().query().join()
    query.all.return_value = "BREADCRUMB"
    get_iso2_to_id_lookup()
    args, kwargs = mocked_reverse_lookup.call_args
    assert (args, kwargs) == (("BREADCRUMB",), {"prefix": "iso_"})


@mock.patch(PATH.format("get_iso2_to_id_lookup"))
@mock.patch(PATH.format("get_arxiv_eu_insts"))
def test_get_arxiv_geo_lookup(mocked_get_insts, mocked_get_iso2):
    mocked_get_iso2.return_value = {"GB": "Something else"}
    mocked_get_insts.return_value = [
        ("58VE", 51.400, -0.1095),
        ("Potterow", 55.9462, -3.1872),
    ]

    assert get_arxiv_geo_lookup() is get_arxiv_geo_lookup()  # ie cache is working
    assert get_arxiv_geo_lookup() == {
        "UK": {"58VE", "Potterow"},
        "UKI": {"58VE"},
        "UKI6": {"58VE"},
        "UKI62": {"58VE"},
        "UKM": {"Potterow"},
        "UKM7": {"Potterow"},
        "UKM75": {"Potterow"},
        "GB": "Something else",
    }


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


@mock.patch(PATH.format("_get_arxiv_articles"))
def test_get_arxiv_articles(mocked_articles):
    output = ["1", "two", "THREE"]
    mocked_articles.return_value = output
    assert next(get_arxiv_articles()) == output


@mock.patch(PATH.format("_get_arxiv_articles"))
@mock.patch(PATH.format("get_arxiv_geo_lookup"))
def test_get_arxiv_articles_geo_split(mocked_lookup, mocked_articles):
    mocked_articles.return_value = [{"id": "1"}, {"id": "two"}, {"id": "THREE"}]
    mocked_lookup.return_value = {"FR": {"1", "THREE"}, "DE": {"two", "1"}}
    assert list(get_arxiv_articles(geo_split=True)) == [
        ([True, False, True], "FR"),
        ([True, True, False], "DE"),
    ]


@mock.patch(PATH.format("get_arxiv_articles"))
@mock.patch(PATH.format("vectorise_docs"), return_value=(None, None))
@mock.patch(PATH.format("fit_topics"), return_value="topic_model")
def test_fit_arxiv_topics(mocked_fit, mocked_vec, mocked_get_articles):
    articles = [{"text": "some text", "title": "a title"}]
    mocked_get_articles().__next__.return_value = articles
    assert fit_arxiv_topics() == (articles, "topic_model")
