from pathlib import Path

path_to_here = Path(__file__).resolve().parent
mysqldb_path = str(path_to_here / "mysqldb.config")
