# Visual Linear Regression

## Purpose

A visual, intuition-first introduction to linear regression and least squares, built for a HumbleBeeAI Academy session. Learners draw a line by hand, discover why errors are squared, watch the loss surface, and only then meet `scikit-learn` — by which point `model.fit()` holds no mystery.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Notebook

The main teaching notebook is instructor-led: short explanations, "Ask the class" prompts, and collapsible `<details>` blocks that hide implementation code until you want to show it.

```bash
jupyter lab
```

Then open `notebooks/01_visual_linear_regression.ipynb`.

There's also a **student workbook** — the same lesson, same charts, same order, but the core calculations (predictions, residuals, SSE, manual least-squares, sklearn fit/predict, ...) are left as small `TODO` exercises with a collapsed solution right after, so students can code along in class and keep a working copy afterward. Not graded — open `notebooks/02_student_workbook.ipynb`.

## Streamlit

An interactive playground that mirrors the notebook's progression (line → errors → loss landscape → plane → many features). Good for projecting during class.

```bash
streamlit run streamlit_app/app.py
```

## Repository structure

```text
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_visual_linear_regression.ipynb   # instructor-led lesson
│   └── 02_student_workbook.ipynb           # same lesson, fill-in-the-blank exercises
│
├── streamlit_app/
│   └── app.py                              # interactive playground (5 tabs)
│
├── data/
│   ├── study_scores_1d.csv                 # hours_studied -> exam_score
│   └── study_scores_2d.csv                 # hours_studied, practice_problems -> exam_score
│
└── assets/
    └── README.md
```
