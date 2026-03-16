from pathlib import Path
import site
import sys

USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend import database


if __name__ == "__main__":
    database.init_database()
    print("Database initialized.")
