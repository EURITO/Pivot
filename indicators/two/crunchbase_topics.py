# %%

from functools import lru_cache
from indicators.core.nlp_utils import fit_topics, vectorise_docs, join_text
from nesta.core.orms.crunchbase_orm import Organization
from nesta.core.orms.orm_utils import get_mysql_engine, db_session
from googletrans import Translator


FOREIGN_STOPS = set(["de", "en,", "la", "que", "un", "para", "con", "una", "et", "es", "el", "des",
                     "les", "und", "le", "de", "los", "pour""une", "las", "por", "dans", "est",
                     "del", "como", "coro", "más", "du", "se", "nos", "der", "für"])


@lru_cache()
def get_crunchbase_orgs():
    """Get all Crunchbase Organizations

    Returns:
        orgs (list): List of Cruncbase Organization data.
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(Organization.id,
                              Organization.short_description,
                              Organization.long_description,
                              Organization.total_funding_usd,
                              Organization.name,
                              Organization.founded_on)
        return [dict(id=id, text=join_text(long_text, short_text),
                     funding=funding, name=name, founded=founded)
                for id, long_text, short_text, name, funding, founded in query.all()
                if (short_text is not None) and (long_text is not None)]


def fit_crunchbase_topics(n_topics=150):
    orgs = get_crunchbase_orgs()

    doc_vectors, feature_names = vectorise_docs([o['text'] for o in orgs],
                                                extra_stops=FOREIGN_STOPS)
    titles = [o['name'] for o in orgs]
    anchors = [['covid', 'covid_19', "coronavirus", '2019_ncov', 'sars_cov_2']]
    topic_model = fit_topics(dataset_label="crunchbase", doc_vectors=doc_vectors,
                             feature_names=feature_names, titles=titles,
                             n_topics=n_topics, anchors=anchors, anchor_strength=100)
    return orgs, topic_model
