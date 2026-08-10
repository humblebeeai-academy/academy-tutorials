# HumbleBeeAI Academy — Tutorials

A collection of hands-on tutorial sessions for HumbleBeeAI Academy. Each tutorial lives in its own top-level folder and is self-contained (own `README.md`, own `requirements.txt`, own data/notebooks/app).

## Tutorials

| Folder | Topic |
| --- | --- |
| [`linear-regression/`](linear-regression/) | Visual, intuition-first introduction to linear regression and least squares — notebook + Streamlit playground. |

## Adding a new tutorial

1. Create a new top-level folder named after the topic (e.g. `decision-trees/`).
2. Keep it self-contained: its own `README.md` with setup/run instructions and its own `requirements.txt`. Use a local `.venv` inside the folder when working on it — the root `.gitignore` already excludes `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, and `.DS_Store` for every folder in this repo, so no per-tutorial `.gitignore` is needed.
3. Add a row to the table above.
