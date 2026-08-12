# Visual Linear Regression

## Purpose

A visual, intuition-first introduction to linear regression and least squares, built for HumbleBeeAI Academy sessions.

**Session 1** — learners draw a line by hand, discover why errors are squared, watch the loss surface, and only then meet `scikit-learn` — by which point `model.fit()` holds no mystery.

**Session 2** — picks up exactly where Session 1 stopped: is the best-fit line actually *good*, and do individual variables really matter? Covers R² (variance explained relative to a mean baseline), coefficient uncertainty and p-values via `statsmodels`, and ends by trying (and failing) to fit ordinary linear regression to a 0/1 outcome — the cliffhanger into Session 3.

**Session 3** — picks up that cliffhanger: the sigmoid function, the logistic regression model, predicted probability vs. predicted class, decision thresholds, a simple confusion matrix, and a two-feature decision boundary.

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

Then open `notebooks/01a_visual_linear_regression.ipynb`.

There's also a **student workbook** — the same Session 1 lesson, same charts, same order, but the core calculations (predictions, residuals, SSE, manual least-squares, sklearn fit/predict, ...) are left as small `TODO` exercises with a collapsed solution right after, so students can code along in class and keep a working copy afterward. Not graded — open `notebooks/01b_student_workbook.ipynb`.

Session 2's notebook, `notebooks/02a_r2_and_statistical_inference.ipynb`, and Session 3's notebook, `notebooks/03a_logistic_regression.ipynb`, are both instructor-led only (no student workbook yet) and follow the same collapsible-code style.

## Streamlit

An interactive playground that mirrors both sessions' progression. Good for projecting during class.

```bash
streamlit run streamlit_app/app.py
```

Tabs 1–5 cover Session 1 (fit a line → understand errors → loss landscape → fit a plane → many features). Tabs 6–7 cover Session 2:

- **R² Playground** — the same line-fitting controls, now showing the mean baseline, regression SSE, and live R² (can go negative — try a deliberately bad line to see why).
- **Sampling Variability** — click "Generate another sample" to draw a fresh synthetic sample and refit with `statsmodels`; a `true slope` toggle (set it to 0) shows how often a truly-zero relationship still produces a "significant" p-value by chance.

Tabs 8–9 cover Session 3:

- **Logistic Regression** — the fitted sigmoid over the pass/fail dataset, with a live decision-threshold slider that recolors points by their current predicted class and updates a live confusion matrix / accuracy readout.
- **Decision Boundary** — the two-feature (`hours_studied` + `practice_problems`) version, showing the straight-line boundary that separates predicted pass from predicted fail.

## Repository structure

```text
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01a_visual_linear_regression.ipynb        # Session 1, instructor-led lesson
│   ├── 01b_student_workbook.ipynb                # Session 1, fill-in-the-blank exercises
│   ├── 02a_r2_and_statistical_inference.ipynb    # Session 2, instructor-led lesson
│   └── 03a_logistic_regression.ipynb             # Session 3, instructor-led lesson
│
├── streamlit_app/
│   └── app.py                              # interactive playground (9 tabs)
│
├── data/
│   ├── study_scores_1d.csv                 # hours_studied -> exam_score
│   ├── study_scores_2d.csv                 # hours_studied, practice_problems -> exam_score
│   ├── salary_regression.csv               # years_experience, projects_completed, coffee_cups -> monthly_salary
│   ├── exam_pass_classification.csv        # hours_studied -> passed (0/1), 10 rows, cliffhanger/recap only
│   ├── exam_pass_classification_full.csv   # hours_studied -> passed (0/1), ~45 rows, main teaching dataset
│   └── exam_pass_2d.csv                    # hours_studied, practice_problems -> passed (0/1)
│
└── assets/
    └── README.md
```
