from unittest import mock
from indicators.two.arxiv_topics import (
    get_arxiv_eu_insts,
    make_reverse_lookup,
    get_iso2_to_id_lookup,
    get_arxiv_geo_lookup,
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
    ids = {"58VE", "Potterow"}

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
