"""
arxiv_topics
============

Topic modelling of the arXiv data.
"""

from functools import lru_cache

from indicators.core.config import EU_COUNTRIES, ARXIV_CONFIG
from indicators.core.db import get_mysql_engine
from indicators.core.nlp_utils import fit_topic_model
from indicators.core.nuts_utils import get_geo_lookup
from nesta.core.orms.arxiv_orm import Article as Art
from nesta.core.orms.arxiv_orm import ArticleInstitute as Link
from nesta.core.orms.grid_orm import Institute as Inst
from nesta.core.orms.orm_utils import db_session


def get_arxiv_eu_insts():
    """Get all institutes in arXiv which are in Europe

    Returns:
       institutes (tuple): (institute_id, latitude, longitude)
    """
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


def get_iso2_to_id():
    """
    Fetch and curate a lookup table of ISO2 code to
    the set of article ids for that ISO2 code.
    """
    engine = get_mysql_engine()
    with db_session(engine) as session:
        q = session.query(Link.article_id, Inst.country_code)
        q = q.join(Link, Link.institute_id == Inst.id, isouter=True)
        return list(q.all())


@lru_cache()
def _get_arxiv_articles(from_date):
    """Get all arXiv articles from a given start date.

    Args:
        from_date (str, optional): Min article creation date.

    Returns:
        articles (list): List of arXiv article data.
    """
    engine = get_mysql_engine()
    with db_session(engine) as session:
        query = session.query(Art.id, Art.abstract, Art.title, Art.created)
        query = query.filter(Art.created >= from_date)
        query = query.filter(Art.abstract.isnot(None))
        articles = [
            dict(id=id, text=abstract, title=title, created=created)
            for id, abstract, title, created in query.all()
        ]
    return articles


def get_arxiv_articles(from_date="2015-01-01", geo_split=False):
    """Get all arXiv articles from a given start date. `geo_split`
    alters the behaviour of the function, such that `geo_split=False`
    will yield an article, whereas `geo_split=False` yields
    a tuple of (indexer, geo_code) where indexer can be used to slice
    `articles`

    Args:
        from_date (str, optional): Min article creation date. Defaults to "2015-01-01".
        geo_split (bool, optional): Alter the behaviour to return an indexer
                                    by geography code. Defaults to False.

    Yields:
        articles (dict): an article object
    """
    articles = _get_arxiv_articles(from_date=from_date)
    if geo_split:
        nuts_to_id_lookup = get_geo_lookup(get_arxiv_eu_insts, get_iso2_to_id)
        for geo_code, ids in nuts_to_id_lookup.items():
            indexes = list(article["id"] in ids for article in articles)
            yield indexes, geo_code
    else:
        yield articles


if __name__ == "__main__":
    fit_topic_model(ARXIV_CONFIG, get_arxiv_articles)
