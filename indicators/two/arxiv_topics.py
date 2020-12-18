# %%

from functools import lru_cache

from sqlalchemy.sql.operators import from_
from indicators.core.nlp_utils import fit_topics, vectorise_docs
from indicators.core.constants import EU_COUNTRIES
from nesta.core.orms.orm_utils import get_mysql_engine, db_session

from nesta.core.orms.arxiv_orm import Article as Art
from nesta.core.orms.arxiv_orm import ArticleInstitute as Link
from nesta.core.orms.grid_orm import Institute as Inst
from nesta.core.orms.orm_utils import get_mysql_engine, db_session
from nuts_finder import NutsFinder as _NutsFinder

from collections import defaultdict

NUTS_FIELDNAME_CONVERSION = {'nuts_code': 'NUTS_ID',
                             'nuts_level': 'LEVL_CODE', 'nuts_name': 'NUTS_NAME'}


@lru_cache
def NutsFinder():
    return _NutsFinder()


def get_nuts_info_lookup():
    nf = NutsFinder()
    lookup = {item['properties']['NUTS_ID']: {'nuts_name': item['properties']['NAME_LATN'],
                                              'nuts_level': item['properties']['LEVL_CODE'],
                                              'nuts_code': item['properties']['NUTS_ID']}
              for item in nf.shapes['features']}
    return lookup


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
    iso_lookup = {}
    with db_session(engine) as session:
        q = session.query(Inst.id, Inst.country_code)
        for id, code in q.all():
            nuts_reverse[f'iso_{code}'].add(id)
            iso_lookup[id] = code
    for id, nuts_info in nuts_lookup.values():
        for nuts_region in nuts_info:
            nuts_id = nuts_region['NUTS_ID']
            nuts_reverse[nuts_id].add(id)
#             region_info = {
#                k: nuts_region[v]
#                for k, v in NUTS_FIELDNAME_CONVERSION.items()
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
    articles = _get_arxiv_articles(from_date=from_date)
    if geo_split:
        nuts_info_lookup = get_nuts_info_lookup()
        nuts_reverse = get_arxiv_geo_lookup()
        for geo_code, ids in nuts_reverse.items():
            _articles = filter(lambda article: article['id'] in ids, articles)
            yield _articles, geo_code, nuts_info_lookup
    else:
        yield articles, None, None


def fit_arxiv_topics(n_topics=150):
    articles = get_arxiv_articles()
    doc_vectors, feature_names = vectorise_docs([a['text'] for a in articles])
    titles = [a['title'] for a in articles]
    anchors = [['covid', 'covid_19', "coronavirus", '2019_ncov', 'sars_cov_2']]
    topic_model = fit_topics(dataset_label="arxiv", doc_vectors=doc_vectors,
                             feature_names=feature_names, titles=titles,
                             n_topics=n_topics, anchors=anchors)
    return articles, topic_model

# %%
