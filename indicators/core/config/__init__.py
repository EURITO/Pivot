from pathlib import Path
import os
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


path_to_here = Path(__file__).resolve().parent
MYSQLDB_PATH = str(path_to_here / "mysqldb.config")
NUTS_EDGE_CASES = load_yaml(path_to_here / "nuts_edge_cases.yaml")
EU_COUNTRIES = load_yaml(path_to_here / "eu_countries.yaml")

os.environ["MYSQLDB"] = MYSQLDB_PATH  # for nesta.get_mysql_engine
