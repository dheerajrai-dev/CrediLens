"""
CrediLens Streamlit Application Root Entrypoint
Run with: streamlit run app.py
"""
import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Execute frontend app
import app.frontend.streamlit_app
