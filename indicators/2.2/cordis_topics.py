# TODO: Counts of NiH projects by topic by region
# TODO: Counts of Cordis projects by topic by region


# %%
from functools import lru_cache

from nesta.core.orms.general_orm import NihProject
from nesta.core.orms.orm_utils import get_mysql_engine, db_session


from indicators.core.nlp_utils import process_documents

# %%


@ lru_cache()
def get_nih_projects():
    """[summary]

    Returns:
        [type]: [description]
    """
    engine = get_mysql_engine("MYSQLDB", "mysqldb", "production")
    with db_session(engine) as session:
        query = session.query(NihProject.application_id, NihProject.phr,
                              NihProject.abstract_text, NihProject.project_title,
                              NihProject.project_start)
        query = query.filter(NihProject.project_start > "2010-01-01")
        return [dict(id=id, text=join_text(phr, abstract), title=title, start_date=start_date)
                for id, phr, abstract, title, start_date in query.all()
                if not ((phr is None) and (abstract is None))]


def join_text(*args):
    return ' '.join(filter(None, args))

# %%
