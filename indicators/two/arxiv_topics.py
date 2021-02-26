"""
arxiv_topics
============

Topic modelling of the arXiv data.
"""

from functools import lru_cache
from itertools import groupby
from operator import itemgetter

from indicators.core.config import EU_COUNTRIES, ARXIV_CONFIG
from indicators.core.db import get_mysql_engine
from indicators.core.nlp_utils import fit_topics, vectorise_docs
from indicators.core.nuts_utils import NutsFinder
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


def make_reverse_lookup(data, key=itemgetter(1), prefix=""):
    """Group an iterable of the form (`id`, `code_object`),
    where a `code` exists somewhere in `code_object` (extracted
    with `key`) such that the output is of the form:

       {code: {id1, id2, id3}}

    You might think of this as being {nuts_code: {article ids}}

    Args:
      data: Iterable of the form (`id`, `code_object`)
      key: Function for extracting `code` from `code_object`
      prefix: Prefix for the code, if required (intended to distinguish
              NUTS from ISO codes) (Default value = "")

    Returns:
       Grouped object of the form {code: {id1, id2, id3}}
    """
    # Look nuts code --> article id
    return {
        f"{prefix}{code}": set(id for id, _ in group)
        for code, group in groupby(sorted(data, key=key), key=key)
        if code is not None  # Not interested in null codes
    }


def get_iso2_to_id_lookup():
    """
    Fetch and curate a lookup table of ISO2 code to
    the set of article ids for that ISO2 code.
    """
    engine = get_mysql_engine()
    with db_session(engine) as session:
        q = session.query(Link.article_id, Inst.country_code)
        q = q.join(Link, Link.institute_id == Inst.id, isouter=True)
        iso2_to_id_lookup = make_reverse_lookup(q.all(), prefix="iso_")
    return iso2_to_id_lookup


@lru_cache()
def get_arxiv_geo_lookup():
    """Generate a lookup table of geography codes (NUTS/ISO2)
    to arXiv article IDs.

    Returns:
        geo_to_id_lookup (dict): Lookup of geography codes to arXiv article IDs.
    """
    # Forward lookup
    nf = NutsFinder()
    id_to_nuts_lookup = {
        id: nf.find(lat=lat, lon=lon) for id, lat, lon in get_arxiv_eu_insts()
    }
    id_nuts = [  # splatten out the nuts IDs, ready for grouping
        (id, info["NUTS_ID"])
        for id, nuts_info in id_to_nuts_lookup.items()
        for info in nuts_info
    ]
    # Reverse lookups
    nuts_to_id_lookup = make_reverse_lookup(id_nuts)
    iso2_to_id_lookup = get_iso2_to_id_lookup()
    # Combine lookups and return
    nuts_to_id_lookup.update(**iso2_to_id_lookup)
    return nuts_to_id_lookup


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
        print(query)
        print(query.all())
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
        nuts_to_id_lookup = get_arxiv_geo_lookup()
        for geo_code, ids in nuts_to_id_lookup.items():
            indexes = list(article["id"] in ids for article in articles)
            yield indexes, geo_code
    else:
        yield articles


def fit_arxiv_topics():
    """Fit arXiv topics based on hyperparameters specified in 'arxiv.yaml'.

    Returns:
        articles, topic_model: List of articles, and a trained topic model
    """
    articles = next(get_arxiv_articles())  # only one value (a list), so use next
    texts = [art["text"] for art in articles]
    titles = [art["title"] for art in articles]
    # Prepare the data and fit the model
    doc_vectors, feature_names = vectorise_docs(texts)
    topic_model = fit_topics(
        titles=titles,
        doc_vectors=doc_vectors,
        feature_names=feature_names,
        **ARXIV_CONFIG,
    )
    return articles, topic_model
