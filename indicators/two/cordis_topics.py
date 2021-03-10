"""
cordis_topics
============

Topic modelling of the Cordis data.
"""

from functools import lru_cache

from indicators.core.config import CORDIS_CONFIG
from indicators.core.nuts_utils import iso_to_nuts
from indicators.core.db import get_mysql_engine
from nesta.core.orms.cordis_orm import Project
from nesta.core.orms.cordis_orm import Organisation as Org
from nesta.core.orms.cordis_orm import ProjectOrganisation as Link
from nesta.core.orms.orm_utils import db_session

model_config = CORDIS_CONFIG  # Specify the model config here


def get_nuts_to_id():
    """
    Fetch and curate a lookup table of ISO2 code to
    the set of article ids for that ISO2 code.
    """
    return [(rcn, iso_to_nuts(iso)) for rcn, iso in get_iso2_to_id()]


@lru_cache()
def get_iso2_to_id():
    """
    Fetch and curate a lookup table of ISO2 code to
    the set of article ids for that ISO2 code.
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(Link.project_rcn, Org.country_code)
        query = query.join(Org, Link.organization_id == Org.id)
        query = query.filter(Org.country_code != "")
        return list(query.all())


@lru_cache()
def get_objects(from_date):
    """Get all arXiv articles from a given start date.

    Args:
        from_date (str, optional): Min article creation date.

    Returns:
        articles (list): List of arXiv article data.
    """
    engine = get_mysql_engine()
    with db_session(engine) as session:
        query = session.query(
            Project.rcn,
            Project.objective,
            Project.title,
            Project.start_date_code,
            Project.total_cost,
        )
        query = query.filter(Project.start_date_code > from_date)
        return [
            dict(id=rcn, text=text, title=title, created=date, funding=funding)
            for rcn, text, title, date, funding in query.all()
        ]
