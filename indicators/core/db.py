import os
from indicators.core.config import MYSQLDB_PATH
from nesta.core.orms.orm_utils import get_mysql_engine as _get_mysql_engine

os.environ["MYSQLDB"] = MYSQLDB_PATH


def get_mysql_engine():
    """
    Automatically retrieve credentials via config for
    generating the MySQL engine via nesta.core.orms.orm_utils
    """
    return _get_mysql_engine("MYSQLDB", "mysqldb", "production")
