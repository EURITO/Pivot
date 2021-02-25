"""
"""

from collections import defaultdict
from functools import lru_cache
from itertools import groupby

from indicators.core.config import EU_COUNTRIES
from indicators.core.db import get_mysql_engine
from indicators.core.nlp_utils import fit_topics, vectorise_docs
from indicators.core.nuts_utils import NutsFinder
from nesta.core.orms.arxiv_orm import Article as Art
from nesta.core.orms.arxiv_orm import ArticleInstitute as Link
from nesta.core.orms.grid_orm import Institute as Inst
from nesta.core.orms.orm_utils import db_session


def get_arxiv_eu_insts():
    """ """
    engine = get_mysql_engine()
    with db_session(engine) as session:
        # Join institutes to articles via the link table
        q = session.query(Inst.id, Inst.latitude, Inst.longitude)
        q = q.join(Link, Link.institute_id == Inst.id, isouter=True)
        q = q.join(Art, Link.article_id == Art.id, isouter=True)
        # Skip institutes without geo info
        for field in (Inst.id, Art.id, Inst.latitude, Inst.longitude):
            q = q.filter(field.isnot(None))
        # Skip institutes outside of Europe
        q = q.filter(Inst.country_code.in_(EU_COUNTRIES))
        # Group by, in order to deduplicate institutes
        q = q.group_by(Inst.id)
        # Make the request
        return list(q.all())


def make_reverse_lookup(data, key, prefix=""):
    """

    Args:
      data:
      key:
      prefix:  (Default value = "")

    Returns:

    """
    grouped_by_code = groupby(data, key=key)
    # Look nuts code --> article id
    return {
        f"{prefix}{code}": set(id for id, _ in group)
        for code, group in grouped_by_code
        if code is not None
    }


def get_iso2_article_lookup(nuts_lookup):
    """

    Args:
      nuts_lookup:

    Returns:

    """
    engine = get_mysql_engine()
    with db_session(engine) as session:
        q = session.query(Link.article_id, Inst.country_code)
        q = q.join(Link, Link.institute_id == Inst.id, isouter=True)
        iso2_reverse = make_reverse_lookup(
            q.all(), key=lambda id_code: id_code[1], prefix="iso_"
        )
    return iso2_reverse


@lru_cache()
def get_arxiv_geo_lookup():
    """[summary]

    Returns:
        [type]: [description]
    """
    # Forward lookup
    nf = NutsFinder()
    id_to_nuts_lookup = {
        id: nf.find(lat=lat, lon=lon) for id, lat, lon in get_arxiv_eu_insts()
    }
    # Reverse lookups
    nuts_to_id_lookup = make_reverse_lookup(
        id_to_nuts_lookup.items(), key=lambda item: item[1]["NUTS_ID"]
    )
    iso2_to_id_lookup = get_iso2_article_lookup()
    # Combine lookups and return
    nuts_to_id_lookup.update(**iso2_to_id_lookup)
    return nuts_to_id_lookup


@lru_cache()
def _get_arxiv_articles(from_date="2015-01-01"):
    """Get all NiH projects from a given start date.

    Args:
        from_date (str, optional): First project "start date" to consider. Defaults to "2010-01-01".

    Returns:
        projects (list): List of NiH project data.
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(Art.id, Art.abstract, Art.title, Art.created)
        query = query.filter(Art.created > from_date)
        query = query.filter(Art.abstract != None)
        articles = [
            dict(id=id, text=abstract, title=title, created=created)
            for id, abstract, title, created in query.all()
        ]
    return articles


def get_arxiv_articles(from_date="2015-01-01", geo_split=False):
    """[summary]

    Args:
        from_date (str, optional): [description]. Defaults to "2015-01-01".
        geo_split (bool, optional): [description]. Defaults to False.

    Yields:
        [type]: [description]
    """
    articles = _get_arxiv_articles(from_date=from_date)
    if geo_split:
        # nuts_info_lookup = get_nuts_info_lookup()
        nuts_reverse = get_arxiv_geo_lookup()
        for geo_code, ids in nuts_reverse.items():
            indexes = list(article["id"] in ids for article in articles)
            yield indexes, geo_code  # , nuts_info_lookup
    else:
        yield articles


def fit_arxiv_topics(n_topics=150):
    """[summary]

    Args:
        n_topics (int, optional): [description]. Defaults to 150.

    Returns:
        [type]: [description]
    """
    articles = list(get_arxiv_articles())[0]
    doc_vectors, feature_names = vectorise_docs([a["text"] for a in articles])
    titles = [a["title"] for a in articles]
    anchors = [["covid", "covid_19", "coronavirus", "2019_ncov", "sars_cov_2"]]
    topic_model = fit_topics(
        dataset_label="arxiv",
        doc_vectors=doc_vectors,
        feature_names=feature_names,
        titles=titles,
        n_topics=n_topics,
        anchors=anchors,
    )
    return articles, topic_model
