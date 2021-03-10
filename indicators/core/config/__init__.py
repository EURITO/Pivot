from pathlib import Path
import os
import yaml

PATH_TO_HERE = Path(__file__).resolve().parent


def load_yaml(filename):
    with open(PATH_TO_HERE / f"{filename}.yaml") as f:
        return yaml.safe_load(f)


MYSQLDB_PATH = str(PATH_TO_HERE / "mysqldb.config")
NUTS_EDGE_CASES = load_yaml("nuts_edge_cases")
EU_COUNTRIES = load_yaml("eu_countries")
ARXIV_CONFIG = load_yaml("arxiv")
INDICATORS = load_yaml("indicators")

os.environ["MYSQLDB"] = MYSQLDB_PATH  # for nesta.get_mysql_engine
