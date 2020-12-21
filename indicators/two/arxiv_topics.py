"""
"""

from collections import defaultdict
from functools import lru_cache

from indicators.core.constants import EU_COUNTRIES
from indicators.core.nlp_utils import fit_topics, vectorise_docs
from indicators.core.nuts_utils import NutsFinder  # , get_nuts_info_lookup
from nesta.core.orms.arxiv_orm import Article as Art
from nesta.core.orms.arxiv_orm import ArticleInstitute as Link
from nesta.core.orms.grid_orm import Institute as Inst
from nesta.core.orms.orm_utils import db_session, get_mysql_engine


@ lru_cache()
def get_arxiv_geo_lookup():
    """[summary]

    Returns:
        [type]: [description]
    """
    nf = NutsFinder()
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        q = session.query(Inst.id, Inst.latitude, Inst.longitude)
        q = q.join(Link, Link.institute_id == Inst.id, isouter=True)
        q = q.join(Art, Link.article_id == Art.id, isouter=True)
        for field in (Inst.id, Art.id, Inst.latitude, Inst.longitude):
            q = q.filter(field != None)
        q = q.filter(Inst.country_code.in_(EU_COUNTRIES))
        q = q.group_by(Inst.id)
        nuts_lookup = {id: nf.find(lat=lat, lon=lon)
                       for id, lat, lon in q.all()}

    nuts_reverse = defaultdict(set)
    with db_session(engine) as session:
        q = session.query(Inst.id, Inst.country_code)
        for id, code in q.all():
            nuts_reverse[f'iso_{code}'].add(id)
    for id, nuts_info in nuts_lookup.values():
        for nuts_region in nuts_info:
            nuts_id = nuts_region['NUTS_ID']
            nuts_reverse[nuts_id].add(id)
    return nuts_reverse


@ lru_cache()
def _get_arxiv_articles(from_date="2015-01-01"):
    """Get all NiH projects from a given start date.

    Args:
        from_date (str, optional): First project "start date" to consider. Defaults to "2010-01-01".

    Returns:
        projects (list): List of NiH project data.
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(Art.id, Art.abstract,
                              Art.title, Art.created, Link.grid_id)
        query = session.join(Link, Link.article_id == Art.id)
        query = query.filter(Art.created > from_date)
        query = query.filter(Art.abstract != None)
        return query.all()


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
        #nuts_info_lookup = get_nuts_info_lookup()
        nuts_reverse = get_arxiv_geo_lookup()
        for geo_code, ids in nuts_reverse.items():
            _articles = filter(lambda article: article['id'] in ids, articles)
            yield _articles, geo_code  # , nuts_info_lookup
    else:
        yield articles, None  # , None


def fit_arxiv_topics(n_topics=150):
    """[summary]

    Args:
        n_topics (int, optional): [description]. Defaults to 150.

    Returns:
        [type]: [description]
    """
    articles = get_arxiv_articles()
    doc_vectors, feature_names = vectorise_docs([a['text'] for a in articles])
    titles = [a['title'] for a in articles]
    anchors = [['covid', 'covid_19', "coronavirus", '2019_ncov', 'sars_cov_2']]
    topic_model = fit_topics(dataset_label="arxiv", doc_vectors=doc_vectors,
                             feature_names=feature_names, titles=titles,
                             n_topics=n_topics, anchors=anchors)
    return articles, topic_model
