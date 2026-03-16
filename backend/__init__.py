from __future__ import annotations

import site
import sys


USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    sys.path.append(USER_SITE)
