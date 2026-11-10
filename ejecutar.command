#!/bin/bash
# Doble clic en macOS para abrir el programa
cd "$(dirname "$0")"
python3 -m pip install --disable-pip-version-check -q -r requirements.txt
(sleep 6; open http://localhost:8501) &
python3 -m streamlit run main.py --server.headless true --browser.gatherUsageStats false
