"""
Common utility functions for the MLLab project.
"""

from pathlib import Path
import sys
import os
import subprocess

import nbformat
from nbconvert import HTMLExporter

import streamlit as st

def get_repo_root() -> Path:
    """
    Returns the repository root (MLLab directory).
    Assumes this file is in utils/ under the repo root.
    """
    return Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def render_notebook_to_html(notebook_path: Path) -> str:
    """
    Loads a .ipynb notebook and converts it to HTML using nbconvert.
    The result is cached for performance.
    """
    if not notebook_path.exists():
        return f"<p><strong>Notebook not found:</strong> {notebook_path}</p>"

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    html_exporter.exclude_input = False  # keep code cells
    (body, _resources) = html_exporter.from_notebook_node(nb)
    return body


def show_notebook_viewer(notebook_relative_path: str, height: int = 600):
    """
    Streamlit helper to display a notebook in the UI.
    `notebook_relative_path` is relative to repo root, e.g. 'notebooks/HousePricePrediction.ipynb'.
    """
    import streamlit.components.v1 as components

    root = get_repo_root()
    notebook_path = root / notebook_relative_path

    html = render_notebook_to_html(notebook_path)
    components.html(html, height=height, scrolling=True)


def open_notebook_file(notebook_relative_path: str):
    """
    Attempts to open the given notebook using the OS default handler.

    On Windows: uses os.startfile
    On macOS: uses 'open'
    On Linux: uses 'xdg-open'
    """
    root = get_repo_root()
    notebook_path = root / notebook_relative_path

    if not notebook_path.exists():
        st.warning(f"Notebook file not found: {notebook_path}")
        return

    try:
        if sys.platform.startswith("win"):
            os.startfile(notebook_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(notebook_path)])
        else:
            subprocess.Popen(["xdg-open", str(notebook_path)])
    except Exception as e:
        st.error(f"Could not open notebook file: {e}")
