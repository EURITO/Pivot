# %%

from functools import lru_cache
from indicators.core.nlp_utils import fit_topics, vectorise_docs
from nesta.core.orms.arxiv_orm import Article, ArticleTopic
from nesta.core.orms.orm_utils import get_mysql_engine, db_session


@lru_cache()
def get_arxiv_articles(from_date="2015-01-01"):
    """Get all NiH projects from a given start date.

    Args:
        from_date (str, optional): First project "start date" to consider. Defaults to "2010-01-01".

    Returns:
        projects (list): List of NiH project data.
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(Article.id, Article.abstract,
                              Article.title, Article.created)
        query = query.filter(Article.created > from_date)
        return [dict(id=id, text=text, title=title, created=created)
                for id, text, title, created in query.all() if text is not None]


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
