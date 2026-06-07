import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PIPELINE_VERSION

if PIPELINE_VERSION == "v1":
    from app.v1.main import app
elif PIPELINE_VERSION == "v2":
    from app.v2.pipeline.orchestrator import app
else:
    raise ValueError(f"Unknown PIPELINE_VERSION: {PIPELINE_VERSION}. Must be 'v1' or 'v2'.")
