from pathlib import Path

path_to_here = Path(__file__).resolve().parent
MYSQLDB_PATH = str(path_to_here / "mysqldb.config")
