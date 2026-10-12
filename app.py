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

import streamlit.runtime

if __name__ == "__main__" and not streamlit.runtime.exists():
    print("CrediLens Streamlit Frontend Entrypoint")
    print("To launch the Streamlit frontend UI, run:")
    print("    streamlit run app.py")
    print("To launch the FastAPI backend server, run:")
    print("    python main.py")
    sys.exit(0)

# Execute frontend app
import app.frontend.streamlit_app
