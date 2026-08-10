# PLAN.md — Visual Introduction to Linear Regression and Least Squares

## 1. Project Goal

Build a complete, instructor-ready teaching package for an introductory HumbleBeeAI Academy session on **Linear Regression and Least Squares**.

The session should teach linear regression primarily through **intuition, experimentation, and visualization**, before introducing mathematical notation and scikit-learn.

The central learning progression is:

**data → draw a line → make predictions → measure errors → square errors → minimize total error → obtain best-fit line → extend line to a plane → generalize to many dimensions → use scikit-learn**

This should NOT feel like a traditional statistics lecture.

The learner should first experience the problem visually and manually, and only afterward see the mathematical formulation.

---

# 2. Target Audience

Learners should already know:

* basic Python
* NumPy basics
* Pandas basics
* basic plotting
* mean/average
* basic algebra
* basic idea of a derivative

They do NOT need prior machine learning experience.

Do not assume knowledge of:

* matrix calculus
* optimization algorithms
* normal equations
* statistical inference
* hypothesis testing for regression
* multicollinearity
* advanced regression diagnostics

---

# 3. Learning Outcomes

By the end of the session, learners should be able to explain:

1. What problem linear regression solves.
2. What a prediction represents.
3. What slope and intercept mean.
4. What a residual/error is.
5. Why simply summing residuals does not work.
6. Why errors are squared.
7. What SSE — Sum of Squared Errors — represents.
8. Why the best regression line is the line minimizing SSE.
9. How changing slope and intercept changes predictions and SSE.
10. Why fitting a model can be thought of as an optimization problem.
11. How one input variable produces a line.
12. How two input variables produce a plane.
13. How 30–40 input variables conceptually produce a higher-dimensional hyperplane.
14. Why we cannot visualize high-dimensional models even though the mathematical idea remains the same.
15. How to train a basic `LinearRegression` model using scikit-learn.
16. How `model.coef_`, `model.intercept_`, and `model.predict()` relate to the concepts learned manually.

The most important final mental model should be:

```text
INPUT DATA
    ↓
choose model parameters
    ↓
make predictions
    ↓
compare predictions to reality
    ↓
calculate residuals
    ↓
square residuals
    ↓
sum them
    ↓
calculate SSE
    ↓
find parameters producing minimum SSE
    ↓
best-fitting model
```

---

# 4. Teaching Philosophy

The order of explanation is extremely important.

DO NOT start with:

```text
y = β0 + β1x + ε
```

DO NOT start with:

```python
LinearRegression().fit(...)
```

DO NOT begin with a derivation of ordinary least squares.

Instead use:

> "Here are some points. Can we draw a line that represents them reasonably well?"

Then allow learners to physically manipulate the line.

Only introduce formulas after learners understand the problem that the formulas are solving.

The desired progression is:

### Experience first

"What happens if I move this line?"

### Measurement second

"How can I determine whether one line is better than another?"

### Mathematics third

"Let's express this idea formally."

### Library last

"Now let's see how sklearn performs what we have been doing manually."

---

# 5. Deliverables

Create the following repository:

```text
linear-regression-visual/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── 01_visual_linear_regression.ipynb
│
├── streamlit_app/
│   └── app.py
│
├── data/
│   ├── study_scores_1d.csv
│   └── study_scores_2d.csv
│
└── assets/
    └── README.md
```

Do not over-engineer the project.

No package structure is necessary.

No Docker.

No database.

No backend API.

No unnecessary abstractions.

The focus is teaching.

---

# 6. Technology

Use:

* Python
* NumPy
* Pandas
* Plotly
* scikit-learn
* Streamlit
* Jupyter / IPython
* ipywidgets only if genuinely useful inside the notebook

Prefer Plotly for visualizations because students should be able to:

* zoom
* rotate
* hover
* inspect points
* interact with 3D plots

Use matplotlib only where it clearly communicates something better.

---

# 7. Dataset Design

Use intentionally simple synthetic datasets.

Avoid Kaggle datasets or complicated preprocessing.

The purpose is to understand regression, not data cleaning.

---

## Dataset 1 — One Input Variable

File:

```text
data/study_scores_1d.csv
```

Columns:

```text
hours_studied
exam_score
```

Approximately 12–20 observations.

Example relationship:

```text
exam_score ≈ 35 + 5.5 × hours_studied + noise
```

Values should not lie perfectly on a line.

There must be visible residuals.

Approximate range:

```text
hours_studied: 1–9
exam_score: 35–90
```

Use a deterministic random seed.

---

## Dataset 2 — Two Input Variables

File:

```text
data/study_scores_2d.csv
```

Columns:

```text
hours_studied
practice_problems
exam_score
```

Approximately 25–40 observations.

Generate approximately:

```text
exam_score =
30
+ 4.5 × hours_studied
+ 1.5 × practice_problems
+ noise
```

Again, do not make the data perfectly linear.

Keep values understandable.

The point is that students can interpret:

```text
more study hours → higher expected score

more practice problems → higher expected score
```

---

# 8. Main Notebook

Create:

```text
notebooks/01_visual_linear_regression.ipynb
```

The notebook should be designed as an **instructor-led lesson**.

It should contain:

* explanation
* questions to ask students
* visualizations
* tiny exercises
* formulas
* instructor notes
* hidden/collapsible implementation code

Avoid large walls of text.

Use short explanations followed immediately by visuals.

---

# 9. Hidden Code Requirement

A major requirement:

Whenever substantial code is used to produce a teaching visualization, include a collapsible Markdown section immediately after or before the visualization.

Use HTML:

````html
<details>
<summary>Show code</summary>

```python
# code here
````

</details>
```

The instructor should therefore be able to:

1. present the visualization without distracting students with implementation details;
2. expand the block;
3. copy the code;
4. paste it into a new code cell if they want to discuss implementation.

The notebook may contain executed code cells required to render outputs, but the teaching narrative should not be dominated by implementation code.

Code should be readable enough for Academy learners.

Avoid unnecessarily clever Python.

---

# 10. Notebook Structure

---

## SECTION 0 — Title and Learning Question

Notebook title:

# Visual Linear Regression: How Does a Machine Find the Best Line?

Opening question:

> We have information about how many hours students studied and their exam scores. If another student studies 6.5 hours, how could we estimate their score?

Do not mention "least squares" immediately.

---

# SECTION 1 — Start With Data

Load the one-variable dataset.

Display the dataframe.

Then create a Plotly scatter plot:

```text
x = hours studied
y = exam score
```

No regression line yet.

Ask learners:

> Do you see a relationship?

> If someone studies more, what generally seems to happen?

> Could one line summarize this relationship?

---

# SECTION 2 — Human Regression

This should be one of the most important parts of the lesson.

Show the scatter plot.

Ask students to visually imagine a line through it.

Then provide an interactive/manual line.

The line follows:

[
\hat y = mx+b
]

Initially avoid explaining formal regression terminology.

Explain simply:

```text
m = how steep the line is
b = where the line starts
```

Create controls for:

```text
slope m
intercept b
```

Preferred ranges:

```text
m: -2 → 12
b: 0 → 70
```

The plot should update live.

Students should be able to try fitting the line manually.

---

# SECTION 3 — Manual Line-Fitting Challenge

Provide a clear challenge box:

> Your challenge: find a line that visually represents the points as well as possible.

Initially hide SSE.

Students should first use their eyes.

Let them experiment.

Examples to deliberately try:

```text
m = 0
b = 60
```

Ask:

> What's wrong with this?

Then:

```text
m = 10
b = 10
```

Ask again.

Then let them find something approximately reasonable.

Important teaching point:

> Different people may choose slightly different lines. We need an objective way to decide which one is actually better.

This naturally motivates the error function.

---

# SECTION 4 — Predictions

Choose a candidate line.

For example:

[
\hat y = 5x+35
]

Show:

```text
actual data point
predicted point on line
```

For one selected observation clearly label:

```text
Actual score
Predicted score
```

Introduce:

[
\hat y
]

as:

> the model's prediction.

Do not introduce residual terminology yet.

---

# SECTION 5 — The Error

Draw a vertical line between:

```text
actual observation
and
prediction on regression line
```

This vertical line represents prediction error.

Introduce:

[
e_i = y_i-\hat y_i
]

Build a small table:

| Hours | Actual | Prediction | Error |
| ----: | -----: | ---------: | ----: |

Make errors visibly include:

```text
positive errors
negative errors
```

Ask:

> If we want one number describing how bad the entire line is, why can't we just add all these errors?

Demonstrate cancellation.

For example:

```text
+4
-4

sum = 0
```

Yet both predictions were wrong.

---

# SECTION 6 — Squaring the Error

Introduce:

[
e_i^2=(y_i-\hat y_i)^2
]

Update table:

| Hours | Actual | Prediction | Error | Squared Error |

Explain three useful properties:

1. Negative errors become positive.
2. Errors cannot cancel.
3. Large mistakes receive much larger penalties.

Show:

```text
error = 2  → squared error = 4
error = 5  → squared error = 25
error = 10 → squared error = 100
```

Mention briefly:

> This also means least squares can be sensitive to outliers.

Do NOT teach robust regression yet.

---

# SECTION 7 — Sum of Squared Errors

Introduce:

[
SSE=\sum_{i=1}^{n}(y_i-\hat y_i)^2
]

Explain:

> We now have one number representing how badly the line fits all observations.

Interpretation:

```text
large SSE = poor fit
smaller SSE = better fit
```

Return to the manual slope/intercept visualization.

NOW show:

```text
Current SSE: XXXX
```

beside the plot.

---

# SECTION 8 — The Regression Game

Turn line fitting into a short classroom competition.

Interface:

```text
Slope slider
Intercept slider

Current SSE
```

Have students attempt to minimize SSE manually.

Optional UI:

```text
Best manual SSE seen so far
```

Do NOT reveal the optimum immediately.

Suggested instructor challenge:

> Can anyone get SSE below 500?

Then progressively:

```text
below 300?
below 200?
```

Thresholds should be adjusted automatically based on generated dataset if necessary.

This is the key intuition:

> You are training the regression model manually.

---

# SECTION 9 — Visualizing Residuals

Make a clean visualization containing:

* actual observations
* candidate regression line
* vertical residual lines

Every observation should have a vertical segment connecting:

```text
prediction on line
↕
actual observation
```

Hover information should show:

```text
x
actual
prediction
residual
squared residual
```

This visualization should be reusable throughout the notebook.

---

# SECTION 10 — What Are We Actually Searching For?

Now formalize the model:

[
\hat y = mx+b
]

Loss:

[
L(m,b)=\sum_{i=1}^{n}(y_i-(mx_i+b))^2
]

Explain:

The dataset is fixed.

The things we can change are:

```text
m
b
```

So regression is asking:

> Which values of m and b produce the smallest error?

Represent visually:

```text
(m, b)
   ↓
predictions
   ↓
residuals
   ↓
SSE
```

---

# SECTION 11 — Visualize the Loss Landscape

Create a 3D Plotly surface.

Axes:

```text
X axis → slope m
Y axis → intercept b
Z axis → SSE
```

Calculate SSE across a grid of slope/intercept combinations.

Surface should clearly show a valley/bowl.

Allow:

* rotation
* zoom
* hover

Mark the minimum point.

Initially consider hiding the minimum marker and revealing it after discussion.

Ask:

> What would training mean on this surface?

Expected answer:

> Find the lowest point.

Important statement:

> Machine learning frequently turns into an optimization problem: choose parameters that minimize some loss function.

---

# SECTION 12 — Connection to Calculus

Keep this section light.

Explain that at the bottom:

[
\frac{\partial L}{\partial m}=0
]

and:

[
\frac{\partial L}{\partial b}=0
]

Do NOT conduct a long symbolic derivation.

The objective is merely connecting:

```text
derivatives
↓
minimum
↓
model parameters
```

Remind learners that the derivative intuition they learned earlier now has a concrete ML application.

---

# SECTION 13 — Calculate the Best Line Manually

Now show the closed-form simple-regression solution:

[
m=
\frac{
\sum (x_i-\bar{x})(y_i-\bar{y})
}{
\sum (x_i-\bar{x})^2
}
]

and:

[
b=\bar y-m\bar x
]

Important instructor note:

> Students should understand what these formulas FIND. They do not need to memorize the formulas.

Implement the calculation using NumPy in transparent steps.

Avoid doing everything in one line.

Use variables such as:

```python
x_mean
y_mean
numerator
denominator
slope
intercept
```

Print results.

Then calculate SSE.

Compare with the students' manually achieved SSE.

---

# SECTION 14 — Finally Use Scikit-Learn

Only now import:

```python
from sklearn.linear_model import LinearRegression
```

Prepare:

```python
X
y
```

Highlight shape:

```text
X.shape = (n_samples, 1)
y.shape = (n_samples,)
```

Fit:

```python
model = LinearRegression()
model.fit(X, y)
```

Display:

```python
model.coef_
model.intercept_
```

Compare explicitly:

| Method               | Slope | Intercept | SSE |
| -------------------- | ----: | --------: | --: |
| Human attempt        |   ... |       ... | ... |
| Manual least squares |   ... |       ... | ... |
| sklearn              |   ... |       ... | ... |

The manual least-squares and sklearn values should match to numerical precision.

This is the "aha" moment.

Instructor statement:

> `model.fit()` looked like one line of code. But you now know what mathematical problem that one line solved.

---

# SECTION 15 — Prediction

Use:

```text
6.5 study hours
```

Ask students to predict manually from:

[
\hat y=mx+b
]

Then run:

```python
model.predict(...)
```

Compare results.

Make clear:

Training:

```text
finding parameters
```

Prediction:

```text
using those parameters
```

---

# SECTION 16 — Student Practice: One Variable

Give students a second tiny dataset or generate another deterministic dataset.

Tasks:

1. Plot data.
2. Draw/guess regression line.
3. Choose slope and intercept manually.
4. Calculate predictions.
5. Calculate residuals.
6. Calculate squared residuals.
7. Calculate SSE.
8. Fit using sklearn.
9. Compare their SSE with sklearn's SSE.
10. Predict one new observation.

Include empty/partially completed code cells.

Solutions should be placed inside collapsible:

```html
<details>
<summary>Solution</summary>
...
</details>
```

so learners do not immediately see them.

---

# SECTION 17 — The Big Question

After students understand one input ask:

> Real models normally don't only receive one piece of information. What happens if the prediction depends on TWO things?

Example:

Predict exam score using:

```text
hours studied
+
number of practice problems
```

Show dataframe:

| Hours | Problems | Score |
| ----: | -------: | ----: |

Then introduce:

[
\hat y=b+w_1x_1+w_2x_2
]

Explain:

Previously:

```text
one input
↓
one coefficient
↓
line
```

Now:

```text
two inputs
↓
two coefficients
↓
plane
```

---

# SECTION 18 — 3D Regression

Use Plotly `Scatter3d`.

Axes:

```text
X → Hours studied
Y → Practice problems
Z → Exam score
```

First show only points.

Allow learners to rotate the graph.

Ask them:

> Where would you put a flat surface that approximately passes through these points?

Then add a regression plane.

---

# SECTION 19 — Manual Plane Fitting

At this point transition heavily toward the Streamlit application.

Model:

[
\hat y=b+w_1x_1+w_2x_2
]

Controls:

```text
Intercept b
Hours coefficient w1
Practice coefficient w2
```

Changing them should rotate or move the plane.

Learners manually attempt to fit the plane.

Display live:

```text
Current SSE
```

This should feel exactly like the earlier line-fitting exercise.

Core teaching analogy:

```text
1 feature  → adjust a line
2 features → adjust a plane
```

---

# 20. Streamlit Application

Create:

```text
streamlit_app/app.py
```

The app should be polished enough to project during the lesson.

Use a wide layout.

Recommended title:

# Linear Regression Playground

Subtitle:

> Fit the model yourself before letting the computer do it.

---

# 21. Streamlit Navigation

Use either tabs or sidebar navigation.

Preferred tabs:

```text
1. Fit a Line
2. Understand Errors
3. Loss Landscape
4. Fit a Plane
5. Many Features
```

The app should progressively follow the same conceptual sequence as the notebook.

---

# TAB 1 — Fit a Line

Show:

* 1D dataset
* scatter plot
* manually controlled line

Controls:

```text
Slope
Intercept
```

Options/toggles:

```text
Show predictions
Show residuals
Show SSE
Show best-fit line
```

Default:

```text
best-fit line OFF
```

Students should not see the answer immediately.

Display:

```text
Current equation
ŷ = mx + b

Current SSE
```

Optional:

```text
Best SSE achieved during this session
```

Button:

```text
Reset
```

Button:

```text
Reveal best fit
```

When revealed:

* add sklearn line
* display optimal parameters
* compare current line vs optimal line

---

# TAB 2 — Understand Errors

Allow selection of one observation.

Highlight:

```text
actual value
predicted value
residual
squared residual
```

Display calculation dynamically.

Example:

```text
Actual = 68.0
Prediction = 63.4

Residual
68.0 - 63.4 = 4.6

Squared error
4.6² = 21.16
```

Below it show complete dataset error table.

Columns:

```text
x
actual
prediction
residual
squared_error
```

At bottom:

```text
SSE = sum(squared_error)
```

---

# TAB 3 — Loss Landscape

Show 3D surface:

```text
slope
intercept
SSE
```

Use current line parameters to place a marker on the surface.

This is important:

When the user changes `m` and `b`, their point on the loss surface should move.

This creates the direct connection:

```text
changing line
=
moving through parameter space
```

Optional:

show optimal point after clicking:

```text
Show minimum
```

---

# TAB 4 — Fit a Plane

This is the main 2-variable demonstration.

Display 3D scatter.

Controls:

```text
Intercept b
Hours coefficient w1
Practice coefficient w2
```

Render translucent plane.

The learner must be able to:

* rotate plot
* zoom
* hover
* change parameters
* see plane update
* see SSE update

Add optional residual lines from observations vertically to the plane.

Toggle:

```text
Show residuals
```

Button:

```text
Fit with scikit-learn
```

After click display:

```text
Your model:
ŷ = ...

Best model:
ŷ = ...

Your SSE:
...

Best SSE:
...
```

Prefer retaining both planes if visually understandable:

```text
manual plane
best-fit plane
```

Otherwise let students toggle between them.

---

# 22. Critical 3D Teaching Sequence

The Streamlit plane section should support this exact classroom interaction.

### Step 1

Show points only.

Ask:

> What would a model look like here?

### Step 2

Reveal plane.

### Step 3

Set intentionally bad coefficients.

### Step 4

Let students alter the coefficients.

Ask:

> What does w1 seem to control?

> What does w2 seem to control?

### Step 5

Display residuals.

### Step 6

Minimize SSE manually.

### Step 7

Let sklearn reveal the optimum.

---

# TAB 5 — Many Features

This section should explain how the idea scales beyond what humans can visualize.

Start with:

```text
1 feature
```

Equation:

[
\hat y=b+w_1x_1
]

Visual:

```text
LINE
```

Then:

```text
2 features
```

[
\hat y=b+w_1x_1+w_2x_2
]

Visual:

```text
PLANE
```

Then:

```text
3 features
```

[
\hat y=b+w_1x_1+w_2x_2+w_3x_3
]

Explain:

> We are now beyond what we can conveniently visualize because the output adds another dimension.

Then:

```text
40 features
```

[
\hat y=b+\sum_{j=1}^{40}w_jx_j
]

Show a feature matrix.

Example:

| person | x1 | x2 | x3 | ... | x40 |
| ------ | -: | -: | -: | --- | --: |

Label example features:

```text
hours studied
practice problems
attendance
sleep
previous score
...
feature 40
```

Then show:

```text
X.shape = (1000, 40)
```

and:

```text
model.coef_.shape = (40,)
```

Conceptual visualization:

```text
x1 ── w1 ─┐
x2 ── w2 ─┤
x3 ── w3 ─┤
...       ├── SUM + INTERCEPT ──> prediction
x40 ─w40 ─┘
```

Important statement:

> The geometry becomes impossible for us to picture, but the mathematical idea has not changed.

---

# 23. Design Matrix Explanation

Introduce:

[
X=
\begin{bmatrix}
x_{11} & x_{12} & \dots & x_{1p}\
x_{21} & x_{22} & \dots & x_{2p}\
\vdots & \vdots & & \vdots\
x_{n1} & x_{n2} & \dots & x_{np}
\end{bmatrix}
]

Explain:

```text
rows    = observations
columns = features
```

Then:

```text
100 students
40 features

X.shape = (100, 40)
```

The model learns approximately:

```text
40 coefficients + intercept
```

Each observation still produces:

```text
ONE prediction
```

This is the important conceptual answer to:

> "How can 30–40 variables produce one prediction?"

---

# 24. High-Dimensional Mental Model

Provide this diagram in both notebook and Streamlit:

```text
x₁ × w₁
x₂ × w₂
x₃ × w₃
...
x₄₀ × w₄₀
     ↓
    SUM
     +
  intercept
     ↓
 prediction
```

And mathematically:

[
\hat y=b+w_1x_1+w_2x_2+\dots+w_{40}x_{40}
]

Emphasize:

> Linear regression is called linear because these weighted feature contributions are added linearly.

---

# 25. Final sklearn Example

Finish with the 2-feature dataset.

Use:

```python
features = [
    "hours_studied",
    "practice_problems"
]

X = df[features]
y = df["exam_score"]
```

Fit:

```python
model = LinearRegression()
model.fit(X, y)
```

Print coefficients with names.

For example:

```text
Intercept: 31.24

hours_studied       4.41
practice_problems   1.53
```

Do not merely print an array like:

```text
[4.41, 1.53]
```

Build a readable dataframe.

Explain interpretation:

> Holding practice problems constant, one additional study hour is associated with approximately 4.4 additional predicted points in this synthetic example.

Do not over-discuss causal interpretation.

---

# 26. New Prediction With Two Features

Example learner:

```text
hours studied = 6
practice problems = 10
```

Calculate manually:

[
\hat y=b+w_1(6)+w_2(10)
]

Then:

```python
model.predict(...)
```

Show same answer.

Then provide:

```text
30-feature version works exactly the same way.
```

The library simply receives more columns.

---

# 27. Session Timing

Target approximately **110–120 minutes**.

Suggested structure:

### 0–10 minutes

Problem framing and scatter plot.

### 10–25 minutes

Human/manual line drawing.

### 25–40 minutes

Predictions, residuals and squared errors.

### 40–50 minutes

SSE and manual optimization competition.

### 50–60 minutes

Loss landscape + calculus connection.

### 60–70 minutes

Least-squares solution + sklearn.

### 70–80 minutes

1-variable learner practice.

### 80–95 minutes

Two-feature transition and 3D scatter.

### 95–110 minutes

Manual plane fitting in Streamlit.

### 110–115 minutes

30–40 feature generalization.

### 115–120 minutes

Recap.

If the class must fit into 90 minutes:

Skip:

* manual closed-form implementation details
* some student practice
* extended calculus discussion

Do NOT skip:

* manual line fitting
* residuals
* SSE
* loss landscape
* sklearn comparison
* 3D plane
* high-dimensional explanation

---

# 28. Instructor Prompts

Throughout the notebook, add special Markdown callouts labeled:

### Ask the class

Examples:

> Which line looks better? Why?

> Can two wrong predictions cancel each other if we simply add errors?

> What happens to SSE when I deliberately rotate the line away from the points?

> What exactly are we changing when we "train" this model?

> What would happen if we had two inputs instead of one?

> What do you think replaces the line in three dimensions?

> What happens if we have 40 inputs?

Also add:

### Instructor note

These should contain explanations or possible misconceptions.

---

# 29. Misconceptions to Explicitly Address

The materials should directly address these.

### Misconception 1

"The regression line has to pass through every point."

Correct:

It usually will not. It summarizes the overall relationship by minimizing error.

---

### Misconception 2

"The computer draws the line visually."

Correct:

It optimizes numerical parameters.

---

### Misconception 3

"Error means the model made a programming mistake."

Correct:

Residual/error means the difference between observed and predicted values.

---

### Misconception 4

"More features means more outputs."

Correct:

Many features can contribute to a single prediction.

---

### Misconception 5

"Two features mean two regression lines."

Correct:

Two features + one continuous output form a plane in 3D.

---

### Misconception 6

"Linear regression only works with one feature."

Correct:

Linear regression can contain many features.

---

### Misconception 7

"`model.fit()` is magic."

Correct:

It estimates parameters that minimize the regression objective.

---

# 30. What NOT to Teach in This Session

Do not significantly cover:

* R²
* adjusted R²
* train/test split
* cross-validation
* regularization
* Ridge
* Lasso
* polynomial regression
* confidence intervals
* p-values
* coefficient significance testing
* heteroscedasticity
* multicollinearity
* normality assumptions
* feature scaling
* gradient descent implementation
* matrix derivation of OLS
* normal equation derivation
* pseudoinverse
* stochastic gradient descent

These can be mentioned as future material but should not distract from the core objective.

---

# 31. Coding Style

Code should prioritize readability.

Prefer:

```python
predictions = slope * x + intercept
residuals = y - predictions
squared_errors = residuals ** 2
sse = squared_errors.sum()
```

over overly condensed expressions.

Variable names should be educational:

```text
actual
predicted
residual
squared_error
slope
intercept
```

Avoid excessive mathematical Greek variable names in Python.

Use:

```python
slope
intercept
hours_coefficient
practice_coefficient
```

instead of:

```python
β1
θ2
```

Greek notation may be used in displayed mathematics.

---

# 32. Streamlit UX Requirements

The application should look clean on a projector.

Use:

```python
st.set_page_config(
    page_title="Linear Regression Playground",
    layout="wide"
)
```

Where useful use:

```text
left column  → controls
right column → visualization
```

Avoid overcrowding.

Large visualizations are preferable.

The app should behave correctly without requiring page refreshes.

Controls should update plots immediately.

Maintain deterministic generated data.

---

# 33. Plotly Requirements

All major graphs should have:

* meaningful axis names
* titles
* hover information
* clear legends
* sensible axis ranges
* appropriate marker sizes

For 3D:

* make points clearly visible
* use a semi-transparent plane
* allow rotation
* avoid a plane so opaque that points disappear

Residual lines should be visually distinct.

Do not rely exclusively on color to convey meaning.

---

# 34. README

Create a concise `README.md`.

Include:

## Purpose

Visual introduction to linear regression and least squares.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows equivalent may be included.

## Notebook

```bash
jupyter lab
```

## Streamlit

```bash
streamlit run streamlit_app/app.py
```

## Repository structure

Brief explanation of each file.

---

# 35. requirements.txt

At minimum:

```text
numpy
pandas
plotly
scikit-learn
streamlit
jupyter
ipywidgets
```

Use reasonable current package versions only if pinning is necessary.

Do not introduce unnecessary dependencies.

---

# 36. Final Recap Section

The notebook should end with a large summary.

## One Feature

[
\hat y=b+w_1x_1
]

Visual:

```text
LINE
```

---

## Two Features

[
\hat y=b+w_1x_1+w_2x_2
]

Visual:

```text
PLANE
```

---

## Many Features

[
\hat y=b+\sum_{j=1}^{p}w_jx_j
]

Visual:

```text
HYPERPLANE
```

Then:

```text
Prediction
    ↓
Residual
    ↓
Square
    ↓
Sum
    ↓
SSE
    ↓
Minimize
    ↓
Best coefficients
```

Final message:

> Linear regression is not fundamentally about drawing a line. It is about learning coefficients that combine input features to produce predictions while minimizing prediction error. A line is simply the version we are lucky enough to visualize.

---

# 37. Final Student Check

End with five questions.

### Question 1

What is a residual?

### Question 2

Why don't we simply add raw residuals?

### Question 3

What does least squares try to minimize?

### Question 4

If one input produces a line, what do two inputs produce?

### Question 5

If a model has 40 input features, how can it still produce one prediction?

Expected conceptual answer:

> Each feature is multiplied by its learned coefficient, the contributions are added together with the intercept, and the result is one predicted value.

---

# 38. Quality / Acceptance Criteria

The project is complete only if all of these work.

### Notebook

* Opens without errors.
* Runs top-to-bottom.
* Uses deterministic data.
* Starts with intuition rather than formulas.
* Has a functioning manual line-fitting visualization.
* Shows residuals graphically.
* Calculates SSE.
* Contains an SSE loss landscape.
* Calculates simple least squares manually.
* Fits equivalent sklearn model.
* Compares manual and sklearn parameters.
* Contains a learner practice exercise.
* Introduces two-feature regression.
* Shows a 3D scatter and plane.
* Explains 30–40 feature generalization.
* Contains collapsible code/solution blocks.
* Includes instructor prompts.

### Streamlit

* Starts successfully using:

```bash
streamlit run streamlit_app/app.py
```

* Manual line controls work.
* SSE updates immediately.
* Residual visualization works.
* Loss surface works.
* Current parameter position can be related to the loss surface.
* 3D plane responds to all three parameters.
* Plane can be rotated interactively.
* sklearn optimum can be revealed.
* Manual vs sklearn parameters and SSE are compared.
* Many-feature explanation is available.

### Teaching quality

A learner who finishes the session should be able to say:

> "Linear regression chooses coefficients that create predictions and tries to make the squared differences between predictions and actual values as small as possible."

If the implementation accomplishes that clearly, prioritize clarity over adding more functionality.

---

# 39. Implementation Priority

Build in this order.

## Priority 1

Create deterministic datasets.

## Priority 2

Build notebook sections 0–9:

```text
scatter
manual line
prediction
residual
squared error
SSE
```

Test thoroughly.

## Priority 3

Build loss landscape.

## Priority 4

Implement manual least-squares calculation and sklearn comparison.

## Priority 5

Build one-variable practice.

## Priority 6

Build 3D two-variable visualization.

## Priority 7

Build Streamlit manual plane-fitting interaction.

## Priority 8

Build many-feature explanation.

## Priority 9

Polish instructional Markdown and instructor prompts.

## Priority 10

Test entire repo from clean environment.

---

# 40. Guiding Principle for the AI Coding Agent

When deciding whether to add something, ask:

> Does this make it easier for a beginner to SEE what linear regression is doing?

If yes, add it.

If it merely makes the implementation more sophisticated, do not add it.

The strongest parts of this project should be:

1. manually moving a regression line;
2. seeing residuals appear;
3. watching SSE respond;
4. seeing the loss valley;
5. discovering sklearn's optimal line;
6. moving from a line to a 3D plane;
7. realizing that a 40-feature model is conceptually the same operation in a space humans cannot visualize.
