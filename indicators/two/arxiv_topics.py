"""
arxiv_topics
============

Topic modelling of the arXiv data.
"""

from functools import lru_cache
import logging

from indicators.core.config import EU_COUNTRIES, ARXIV_CONFIG
from indicators.core.db import get_mysql_engine
from nesta.core.orms.arxiv_orm import Article as Art
from nesta.core.orms.arxiv_orm import ArticleInstitute as Link
from nesta.core.orms.grid_orm import Institute as Inst
from nesta.core.orms.orm_utils import db_session

model_config = ARXIV_CONFIG  # Specify the model config here


def get_lat_lon():
    """Get all institutes in arXiv which are in Europe

    Returns:
       institutes (tuple): (institute_id, latitude, longitude)
    """
    logging.info("Retrieving articles lat lon lookup")
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
    logging.info("Retrieving articles iso2 lookup")
    engine = get_mysql_engine()
    with db_session(engine) as session:
        q = session.query(Link.article_id, Inst.country_code)
        q = q.join(Link, Link.institute_id == Inst.id, isouter=True)
        for field in (Inst.id, Link.article_id):
            q = q.filter(field.isnot(None))
        return list(q.all())


@lru_cache()
def get_objects(from_date):
    """Get all arXiv articles from a given start date.

    Args:
        from_date (str, optional): Min article creation date.

    Returns:
        articles (list): List of arXiv article data.
    """
    logging.info(f"Retrieving articles from at least '{from_date}'")
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
