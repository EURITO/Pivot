"""
nih_topics
============

Topic modelling of the NIH data.
"""

from functools import lru_cache

from indicators.core.config import NIH_CONFIG
from indicators.core.nlp_utils import join_text
from indicators.core.db import get_mysql_engine
from nesta.core.orms.arxiv_orm import NihProject as Project
from nesta.core.orms.orm_utils import db_session

model_config = NIH_CONFIG  # Specify the model config here


@lru_cache
def get_projects():
    engine = get_mysql_engine()
    with db_session(engine) as session:
        query = session.query(
            Project.application_id, Project.coordinates, Project.is_eu, Project.iso2
        )
        return query.all()


def get_lat_lon():
    """Get all institutes in NIH which are in Europe

    Returns:
       institutes (tuple): (institute_id, latitude, longitude)
    """
    projects = get_projects()
    return [
        (id, float(coords["lat"]), float(coords["lon"]))
        for id, coords, is_eu, _ in projects
        if is_eu and coords is not None
    ]


def get_iso2_to_id():
    """
    Fetch and curate a lookup table of ISO2 code to
    the set of article ids for that ISO2 code.
    """
    projects = get_projects()
    return [(id, iso_code) for id, _, _, iso_code in projects]


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
            Project.application_id,
            Project.phr,
            Project.abstract_text,
            Project.project_title,
            Project.project_start,
            Project.total_cost,
        )
        query = query.filter(Project.project_start > from_date)
        projects = [
            dict(
                id=id,
                text=join_text(phr, abstract),
                title=title,
                start_date=start_date,
                funding=funding,
            )
            for id, phr, abstract, title, start_date, funding in query.all()
            if not ((phr is None) and (abstract is None))
        ]
    return projects
