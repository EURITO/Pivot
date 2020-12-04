# TODO: Counts of NiH projects by topic by region
# TODO: Counts of Cordis projects by topic by region


# %%
from functools import lru_cache
from matplotlib.pyplot import title
from sklearn.feature_extraction.text import CountVectorizer
from nesta.core.orms.general_orm import NihProject
from nesta.core.orms.orm_utils import get_mysql_engine, db_session


from indicators.core.nlp_utils import fit_topics, vectorise_docs


@lru_cache()
def get_nih_projects(from_date="2010-01-01"):
    """Get all NiH projects from a given start date.

    Args:
        from_date (str, optional): First project "start date" to consider. Defaults to "2010-01-01".

    Returns:
        projects (list): List of NiH project data.
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(NihProject.application_id, NihProject.phr,
                              NihProject.abstract_text, NihProject.project_title,
                              NihProject.project_start, NihProject.total_cost)
        query = query.filter(NihProject.project_start > from_date)
        return [dict(id=id, text=join_text(phr, abstract), title=title,
                     start_date=start_date, funding=funding)
                for id, phr, abstract, title, start_date, funding in query.all()
                if not ((phr is None) and (abstract is None))]


def join_text(*args):
    """Concatenate all text arguments together, ignore None values.

    Args:
        args (tuple): Any sequence of str or None values.
    Returns:
        joined: All text values concatenated together with a single space token.
    """
    return ' '.join(filter(None, args))

# %%


def fit_nih_topics(n_topics=150):
    projects = get_nih_projects()
    doc_vectors, feature_names = vectorise_docs([p['text'] for p in projects])
    titles = [p['title'] for p in projects]
    anchors = [['infection', 'covid', "coronavirus"],
               ["infection", "hiv"],
               ["infection", "bacteria"],
               ["infection", "parasite"]]
    topic_model = fit_topics(dataset_label="nih", doc_vectors=doc_vectors,
                             feature_names=feature_names, titles=titles,
                             n_topics=n_topics, anchors=anchors)
    return projects, topic_model
