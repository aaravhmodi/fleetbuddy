"""Entry point: `uvicorn main:app --reload --env-file .env`.

The code lives in the fleet_buddy/ package; start with fleet_buddy/api.py (endpoints)
and fleet_buddy/chat.py (the model/tool loop).
"""

from fleet_buddy.api import app

__all__ = ["app"]
