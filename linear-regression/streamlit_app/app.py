"""Linear Regression Playground — an interactive companion to the notebook.

Run with: streamlit run streamlit_app/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="Linear Regression Playground", layout="wide")


@st.cache_data
def load_1d_data():
    return pd.read_csv(DATA_DIR / "study_scores_1d.csv")


@st.cache_data
def load_2d_data():
    return pd.read_csv(DATA_DIR / "study_scores_2d.csv")


@st.cache_data
def load_classification_data():
    return pd.read_csv(DATA_DIR / "exam_pass_classification_full.csv")


@st.cache_data
def load_classification_2d_data():
    return pd.read_csv(DATA_DIR / "exam_pass_2d.csv")


scores_1d = load_1d_data()
scores_2d = load_2d_data()
pass_df = load_classification_data()
pass_2d_df = load_classification_2d_data()

SLOPE_MIN, SLOPE_MAX = -2.0, 12.0
INTERCEPT_MIN, INTERCEPT_MAX = 0.0, 70.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@st.cache_resource
def fit_1d_model(dataframe):
    model = LinearRegression()
    model.fit(dataframe[["hours_studied"]], dataframe["exam_score"])
    return model


@st.cache_resource
def fit_2d_model(dataframe):
    model = LinearRegression()
    model.fit(dataframe[["hours_studied", "practice_problems"]], dataframe["exam_score"])
    return model


@st.cache_resource
def fit_logistic_model(dataframe):
    model = LogisticRegression()
    model.fit(dataframe[["hours_studied"]], dataframe["passed"])
    return model


@st.cache_resource
def fit_logistic_model_2d(dataframe):
    model = LogisticRegression()
    model.fit(dataframe[["hours_studied", "practice_problems"]], dataframe["passed"])
    return model


best_1d_model = fit_1d_model(scores_1d)
best_2d_model = fit_2d_model(scores_2d)
logistic_model = fit_logistic_model(pass_df)
logistic_model_2d = fit_logistic_model_2d(pass_2d_df)


def compute_predictions_1d(dataframe, slope, intercept):
    return slope * dataframe["hours_studied"] + intercept


def compute_sse_1d(dataframe, slope, intercept):
    residuals = dataframe["exam_score"] - compute_predictions_1d(dataframe, slope, intercept)
    return float((residuals ** 2).sum())


def compute_predictions_2d(dataframe, intercept, w1, w2):
    return intercept + w1 * dataframe["hours_studied"] + w2 * dataframe["practice_problems"]


def compute_sse_2d(dataframe, intercept, w1, w2):
    residuals = dataframe["exam_score"] - compute_predictions_2d(dataframe, intercept, w1, w2)
    return float((residuals ** 2).sum())


# session state defaults
st.session_state.setdefault("slope", 5.0)
st.session_state.setdefault("intercept", 35.0)
st.session_state.setdefault("best_sse_seen", None)
st.session_state.setdefault("reveal_best_line", False)
st.session_state.setdefault("plane_intercept", 30.0)
st.session_state.setdefault("plane_w1", 4.0)
st.session_state.setdefault("plane_w2", 1.0)
st.session_state.setdefault("reveal_best_plane", False)
st.session_state.setdefault("r2_slope", 5.0)
st.session_state.setdefault("r2_intercept", 35.0)
st.session_state.setdefault("sampling_history", [])
st.session_state.setdefault("sampling_true_slope", 5.0)
st.session_state.setdefault("classification_threshold", 0.5)

st.title("Linear Regression Playground")
st.caption("Fit the model yourself before letting the computer do it.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "1. Fit a Line", "2. Understand Errors", "3. Loss Landscape", "4. Fit a Plane", "5. Many Features",
        "6. R² Playground", "7. Sampling Variability",
        "8. Logistic Regression", "9. Decision Boundary",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1 — Fit a Line
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Fit a Line")
    st.write("Adjust the slope and intercept to make your line represent the data as well as you can.")

    control_col, plot_col = st.columns([1, 2])

    with control_col:
        slope = st.slider("Slope (m)", SLOPE_MIN, SLOPE_MAX, st.session_state["slope"], 0.1, key="slope")
        intercept = st.slider("Intercept (b)", INTERCEPT_MIN, INTERCEPT_MAX, st.session_state["intercept"], 0.5, key="intercept")

        show_predictions = st.checkbox("Show predictions", value=False)
        show_residuals = st.checkbox("Show residuals", value=False)
        show_sse = st.checkbox("Show SSE", value=True)
        show_best_fit = st.checkbox("Show best-fit line", value=False)

        current_sse = compute_sse_1d(scores_1d, slope, intercept)
        if st.session_state["best_sse_seen"] is None or current_sse < st.session_state["best_sse_seen"]:
            st.session_state["best_sse_seen"] = current_sse

        st.markdown(f"**Current equation:** ŷ = {slope:.1f}x + {intercept:.1f}")
        if show_sse:
            st.markdown(f"**Current SSE:** {current_sse:.1f}")
        st.markdown(f"Best SSE achieved this session: **{st.session_state['best_sse_seen']:.1f}**")

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if st.button("Reset"):
                st.session_state["slope"] = 5.0
                st.session_state["intercept"] = 35.0
                st.session_state["best_sse_seen"] = None
                st.session_state["reveal_best_line"] = False
                st.rerun()
        with button_col2:
            if st.button("Reveal best fit"):
                st.session_state["reveal_best_line"] = True

        st.markdown("---")
        st.markdown("**Predict for a new student**")
        new_x_hours = st.number_input(
            "Hours studied (not in the dataset)", min_value=0.0, max_value=15.0, value=6.5, step=0.5
        )
        new_y_prediction = slope * new_x_hours + intercept
        st.markdown(f"Predicted score: **{new_y_prediction:.1f}**")

    with plot_col:
        predictions = compute_predictions_1d(scores_1d, slope, intercept)
        residuals = scores_1d["exam_score"] - predictions

        line_x_min = min(scores_1d["hours_studied"].min(), new_x_hours) - 0.5
        line_x_max = max(scores_1d["hours_studied"].max(), new_x_hours) + 0.5
        line_x = np.linspace(line_x_min, line_x_max, 50)

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=scores_1d["hours_studied"], y=scores_1d["exam_score"], mode="markers",
                marker=dict(size=11, color="#2563eb"), name="actual",
                hovertemplate="Hours: %{x}<br>Actual: %{y}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=line_x, y=slope * line_x + intercept, mode="lines",
                line=dict(color="#dc2626", width=3), name=f"y = {slope:.1f}x + {intercept:.1f}",
            )
        )

        if show_predictions:
            figure.add_trace(
                go.Scatter(
                    x=scores_1d["hours_studied"], y=predictions, mode="markers",
                    marker=dict(size=9, color="#dc2626", symbol="x"), name="prediction",
                )
            )

        if show_residuals:
            for xi, yi, pi in zip(scores_1d["hours_studied"], scores_1d["exam_score"], predictions):
                figure.add_trace(
                    go.Scatter(
                        x=[xi, xi], y=[pi, yi], mode="lines",
                        line=dict(color="#f59e0b", width=2, dash="dot"), showlegend=False,
                    )
                )

        if show_best_fit or st.session_state["reveal_best_line"]:
            best_slope = best_1d_model.coef_[0]
            best_intercept = best_1d_model.intercept_
            figure.add_trace(
                go.Scatter(
                    x=line_x, y=best_slope * line_x + best_intercept, mode="lines",
                    line=dict(color="#16a34a", width=3, dash="dash"),
                    name=f"best fit: y = {best_slope:.1f}x + {best_intercept:.1f}",
                )
            )

        # new/unseen prediction: dashed guide lines up to the line, then across to the axis
        figure.add_trace(
            go.Scatter(
                x=[new_x_hours, new_x_hours], y=[0, new_y_prediction], mode="lines",
                line=dict(color="#7c3aed", width=2, dash="dash"), showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0, new_x_hours], y=[new_y_prediction, new_y_prediction], mode="lines",
                line=dict(color="#7c3aed", width=2, dash="dash"), showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[new_x_hours], y=[new_y_prediction], mode="markers",
                marker=dict(size=14, color="#7c3aed", symbol="star"),
                name="new prediction (no actual value)",
                hovertemplate=f"x: {new_x_hours} (new / unseen)<br>predicted y: {new_y_prediction:.1f}<extra></extra>",
            )
        )

        figure.update_layout(
            title="Hours Studied vs Exam Score", xaxis_title="Hours studied", yaxis_title="Exam score",
            template="plotly_white", height=550,
        )
        st.plotly_chart(figure, use_container_width=True)

        if st.session_state["reveal_best_line"]:
            best_slope = best_1d_model.coef_[0]
            best_intercept = best_1d_model.intercept_
            best_sse = compute_sse_1d(scores_1d, best_slope, best_intercept)
            st.info(
                f"**Optimal line:** ŷ = {best_slope:.2f}x + {best_intercept:.2f} "
                f"(SSE = {best_sse:.1f}) — your line: SSE = {current_sse:.1f}"
            )

# ---------------------------------------------------------------------------
# TAB 2 — Understand Errors
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Understand Errors")
    st.write("Pick one student and see exactly how their error is calculated.")

    error_table = scores_1d.copy()
    error_table["prediction"] = compute_predictions_1d(scores_1d, slope, intercept)
    error_table["residual"] = error_table["exam_score"] - error_table["prediction"]
    error_table["squared_error"] = error_table["residual"] ** 2

    selected_index = st.selectbox(
        "Select an observation",
        options=error_table.index,
        format_func=lambda i: f"Student {i}: {error_table.loc[i, 'hours_studied']} hours studied",
    )
    row = error_table.loc[selected_index]

    st.markdown(f"""
Using the current line from Tab 1 (ŷ = {slope:.1f}x + {intercept:.1f}):

```text
Actual     = {row['exam_score']:.1f}
Prediction = {row['prediction']:.1f}

Residual
{row['exam_score']:.1f} - {row['prediction']:.1f} = {row['residual']:.2f}

Squared error
{row['residual']:.2f}^2 = {row['squared_error']:.2f}
```
""")

    st.write("Full dataset error table:")
    st.dataframe(
        error_table[["hours_studied", "exam_score", "prediction", "residual", "squared_error"]]
        .rename(columns={"hours_studied": "x", "exam_score": "actual"})
        .round(2),
        use_container_width=True,
    )
    st.markdown(f"**SSE = sum(squared_error) = {error_table['squared_error'].sum():.1f}**")

# ---------------------------------------------------------------------------
# TAB 3 — Loss Landscape
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Loss Landscape")
    st.write("Every point on this surface is one possible (slope, intercept) pair and its resulting SSE.")

    show_minimum = st.checkbox("Show minimum", value=False)

    slope_grid_vals = np.linspace(SLOPE_MIN, SLOPE_MAX, 50)
    intercept_grid_vals = np.linspace(INTERCEPT_MIN, INTERCEPT_MAX, 50)
    slope_grid, intercept_grid = np.meshgrid(slope_grid_vals, intercept_grid_vals)

    x_values = scores_1d["hours_studied"].to_numpy()
    y_values = scores_1d["exam_score"].to_numpy()
    predictions_grid = slope_grid[..., None] * x_values + intercept_grid[..., None]
    sse_grid = ((y_values - predictions_grid) ** 2).sum(axis=-1)

    surface_figure = go.Figure(
        data=[go.Surface(x=slope_grid, y=intercept_grid, z=sse_grid, colorscale="Viridis", opacity=0.85)]
    )

    current_marker_sse = compute_sse_1d(scores_1d, slope, intercept)
    surface_figure.add_trace(
        go.Scatter3d(
            x=[slope], y=[intercept], z=[current_marker_sse],
            mode="markers", marker=dict(size=6, color="#dc2626"), name="your line",
        )
    )

    if show_minimum:
        best_index = np.unravel_index(np.argmin(sse_grid), sse_grid.shape)
        surface_figure.add_trace(
            go.Scatter3d(
                x=[slope_grid[best_index]], y=[intercept_grid[best_index]], z=[sse_grid[best_index]],
                mode="markers", marker=dict(size=6, color="#16a34a"), name="minimum",
            )
        )

    surface_figure.update_layout(
        scene=dict(xaxis_title="slope (m)", yaxis_title="intercept (b)", zaxis_title="SSE"),
        height=650, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(surface_figure, use_container_width=True)
    st.caption("Moving the sliders on Tab 1 moves the red marker on this surface.")

# ---------------------------------------------------------------------------
# TAB 4 — Fit a Plane
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Fit a Plane")
    st.write("Two features now: hours studied and practice problems. Adjust three numbers to fit a plane.")

    control_col, plot_col = st.columns([1, 2])

    with control_col:
        plane_intercept = st.slider("Intercept (b)", 0.0, 60.0, st.session_state["plane_intercept"], 0.5, key="plane_intercept")
        plane_w1 = st.slider("Hours coefficient (w1)", -2.0, 12.0, st.session_state["plane_w1"], 0.1, key="plane_w1")
        plane_w2 = st.slider("Practice coefficient (w2)", -2.0, 6.0, st.session_state["plane_w2"], 0.1, key="plane_w2")

        show_plane_residuals = st.checkbox("Show residuals", value=False, key="show_plane_residuals")

        plane_sse = compute_sse_2d(scores_2d, plane_intercept, plane_w1, plane_w2)
        st.markdown(f"**Your model:** ŷ = {plane_intercept:.1f} + {plane_w1:.1f}·x1 + {plane_w2:.1f}·x2")
        st.markdown(f"**Your SSE:** {plane_sse:.1f}")

        if st.button("Fit with scikit-learn"):
            st.session_state["reveal_best_plane"] = True

        if st.session_state["reveal_best_plane"]:
            best_intercept = best_2d_model.intercept_
            best_w1, best_w2 = best_2d_model.coef_
            best_sse = compute_sse_2d(scores_2d, best_intercept, best_w1, best_w2)
            st.markdown("---")
            st.markdown(f"**Best model:** ŷ = {best_intercept:.2f} + {best_w1:.2f}·x1 + {best_w2:.2f}·x2")
            st.markdown(f"**Best SSE:** {best_sse:.1f}")

        st.markdown("---")
        st.markdown("**Predict for a new student**")
        new_plane_hours = st.number_input("Hours studied", min_value=0.0, max_value=15.0, value=6.0, step=0.5)
        new_plane_problems = st.number_input("Practice problems", min_value=0.0, max_value=25.0, value=10.0, step=1.0)
        new_plane_prediction = plane_intercept + plane_w1 * new_plane_hours + plane_w2 * new_plane_problems
        st.markdown(f"Predicted score: **{new_plane_prediction:.1f}**")

    with plot_col:
        plane_figure = go.Figure()
        plane_figure.add_trace(
            go.Scatter3d(
                x=scores_2d["hours_studied"], y=scores_2d["practice_problems"], z=scores_2d["exam_score"],
                mode="markers", marker=dict(size=5, color="#2563eb"), name="actual",
                hovertemplate="Hours: %{x}<br>Problems: %{y}<br>Score: %{z}<extra></extra>",
            )
        )

        plane_x_min = min(scores_2d["hours_studied"].min(), new_plane_hours)
        plane_x_max = max(scores_2d["hours_studied"].max(), new_plane_hours)
        plane_y_min = min(scores_2d["practice_problems"].min(), new_plane_problems)
        plane_y_max = max(scores_2d["practice_problems"].max(), new_plane_problems)
        x_range = np.linspace(plane_x_min, plane_x_max, 15)
        y_range = np.linspace(plane_y_min, plane_y_max, 15)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_grid = plane_intercept + plane_w1 * x_grid + plane_w2 * y_grid

        plane_figure.add_trace(
            go.Surface(x=x_grid, y=y_grid, z=z_grid, opacity=0.5, colorscale="Reds", showscale=False, name="your plane")
        )

        if st.session_state["reveal_best_plane"]:
            best_intercept = best_2d_model.intercept_
            best_w1, best_w2 = best_2d_model.coef_
            best_z_grid = best_intercept + best_w1 * x_grid + best_w2 * y_grid
            plane_figure.add_trace(
                go.Surface(x=x_grid, y=y_grid, z=best_z_grid, opacity=0.35, colorscale="Greens", showscale=False, name="best plane")
            )

        if show_plane_residuals:
            plane_predictions = compute_predictions_2d(scores_2d, plane_intercept, plane_w1, plane_w2)
            for xi, yi, zi, pi in zip(
                scores_2d["hours_studied"], scores_2d["practice_problems"], scores_2d["exam_score"], plane_predictions
            ):
                plane_figure.add_trace(
                    go.Scatter3d(
                        x=[xi, xi], y=[yi, yi], z=[pi, zi], mode="lines",
                        line=dict(color="#f59e0b", width=3), showlegend=False,
                    )
                )

        plane_figure.add_trace(
            go.Scatter3d(
                x=[new_plane_hours], y=[new_plane_problems], z=[new_plane_prediction],
                mode="markers", marker=dict(size=7, color="#7c3aed", symbol="diamond"),
                name="new prediction (no actual value)",
                hovertemplate=(
                    f"Hours: {new_plane_hours} (new/unseen)<br>Problems: {new_plane_problems}"
                    f"<br>predicted score: {new_plane_prediction:.1f}<extra></extra>"
                ),
            )
        )

        plane_figure.update_layout(
            scene=dict(xaxis_title="Hours studied", yaxis_title="Practice problems", zaxis_title="Exam score"),
            height=650, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(plane_figure, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 5 — Many Features
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Many Features")
    st.write("The idea scales far beyond what we can draw. The math never changes — only the number of terms.")

    st.markdown("""
| Features | Equation | Visual |
|---|---|---|
| 1 | ŷ = b + w1·x1 | LINE |
| 2 | ŷ = b + w1·x1 + w2·x2 | PLANE |
| 3 | ŷ = b + w1·x1 + w2·x2 + w3·x3 | beyond what we can draw |
| 40 | ŷ = b + Σ(j=1 to 40) wj·xj | impossible to picture, same idea |
""")

    st.markdown("### Example feature matrix")

    example_feature_names = [
        "hours_studied", "practice_problems", "attendance", "sleep_hours", "previous_score", "...", "feature_40",
    ]
    rng = np.random.default_rng(seed=1)
    example_matrix = pd.DataFrame(
        rng.normal(size=(5, len(example_feature_names) - 1)).round(2),
        columns=[name for name in example_feature_names if name != "..."],
    )
    example_matrix.insert(5, "...", "...")
    example_matrix.index = [f"student_{i}" for i in range(1, 6)]
    st.dataframe(example_matrix, use_container_width=True)

    st.markdown("""
```text
X.shape           = (1000, 40)
model.coef_.shape = (40,)
```
""")

    st.markdown("### The mental model")
    st.code(
        "x1  * w1\n"
        "x2  * w2\n"
        "x3  * w3\n"
        "...\n"
        "x40 * w40\n"
        "      |\n"
        "     SUM\n"
        "      +\n"
        "  intercept\n"
        "      |\n"
        "  prediction",
        language="text",
    )

    st.info(
        "Linear regression is called **linear** because these weighted feature contributions "
        "are added together linearly. The geometry becomes impossible to picture past 2-3 features, "
        "but the mathematical idea has not changed."
    )

# ---------------------------------------------------------------------------
# TAB 6 — R² Playground
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("R² Playground")
    st.write(
        "R² compares your line's error against the simplest possible baseline: always predicting the mean. "
        "Move the slope and intercept and watch R² respond."
    )

    r2_control_col, r2_plot_col = st.columns([1, 2])

    baseline_prediction_r2 = scores_1d["exam_score"].mean()
    sst_r2 = float(((scores_1d["exam_score"] - baseline_prediction_r2) ** 2).sum())

    with r2_control_col:
        r2_slope = st.slider("Slope (m)", SLOPE_MIN, SLOPE_MAX, st.session_state["r2_slope"], 0.1, key="r2_slope")
        r2_intercept = st.slider(
            "Intercept (b)", INTERCEPT_MIN, INTERCEPT_MAX, st.session_state["r2_intercept"], 0.5, key="r2_intercept"
        )

        r2_sse = compute_sse_1d(scores_1d, r2_slope, r2_intercept)
        r2_value = 1 - r2_sse / sst_r2

        st.markdown(f"**Baseline SSE (= SST):** {sst_r2:.1f}")
        st.markdown(f"**Your regression SSE:** {r2_sse:.1f}")
        st.markdown(f"**R² = 1 − SSE/SST = {r2_value:.3f}**")

        if r2_value < 0:
            st.warning(
                "R² is negative! That means this line is doing *worse* than simply predicting the "
                "mean for every student — try a value closer to the earlier best-fit line to see R² recover."
            )
        elif r2_value > 0.8:
            st.success("Strong fit — this line explains most of the variation relative to the mean baseline.")

    with r2_plot_col:
        r2_line_x = np.linspace(scores_1d["hours_studied"].min() - 0.5, scores_1d["hours_studied"].max() + 0.5, 50)
        r2_predictions = compute_predictions_1d(scores_1d, r2_slope, r2_intercept)

        r2_figure = go.Figure()
        r2_figure.add_trace(
            go.Scatter(
                x=scores_1d["hours_studied"], y=scores_1d["exam_score"], mode="markers",
                marker=dict(size=11, color="#2563eb"), name="actual",
            )
        )
        r2_figure.add_hline(y=baseline_prediction_r2, line=dict(color="#f59e0b", width=3, dash="dash"))
        r2_figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines", line=dict(color="#f59e0b", width=3, dash="dash"),
                name=f"baseline (mean = {baseline_prediction_r2:.1f})",
            )
        )
        r2_figure.add_trace(
            go.Scatter(
                x=r2_line_x, y=r2_slope * r2_line_x + r2_intercept, mode="lines",
                line=dict(color="#dc2626", width=3), name=f"your line: y = {r2_slope:.1f}x + {r2_intercept:.1f}",
            )
        )
        for xi, yi, pi in zip(scores_1d["hours_studied"], scores_1d["exam_score"], r2_predictions):
            r2_figure.add_trace(
                go.Scatter(
                    x=[xi, xi], y=[pi, yi], mode="lines",
                    line=dict(color="#16a34a", width=1.5, dash="dot"), showlegend=False,
                )
            )
        r2_figure.update_layout(
            title=f"R² = {r2_value:.3f}",
            xaxis_title="Hours studied", yaxis_title="Exam score",
            template="plotly_white", height=550,
        )
        st.plotly_chart(r2_figure, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 7 — Sampling Variability
# ---------------------------------------------------------------------------
with tab7:
    st.subheader("Sampling Variability")
    st.write(
        "Our dataset is only one sample. Every time you generate a new one from the same underlying "
        "relationship, the fitted slope and its p-value come out a little different."
    )

    sampling_control_col, sampling_plot_col = st.columns([1, 2])

    with sampling_control_col:
        true_slope = st.slider(
            "True slope (the real, unknown relationship)", 0.0, 8.0, st.session_state["sampling_true_slope"], 0.5,
            key="sampling_true_slope",
            help="Set this to 0 to simulate a feature that truly has no effect, and watch how often you still get a 'significant'-looking slope by chance.",
        )

        if st.button("Generate another sample"):
            sample_rng = np.random.default_rng()
            sample_x = sample_rng.uniform(1, 9, size=30)
            sample_y = 40 + true_slope * sample_x + sample_rng.normal(loc=0, scale=10, size=30)

            sample_X_sm = sm.add_constant(sample_x)
            sample_results = sm.OLS(sample_y, sample_X_sm).fit()
            sample_slope = sample_results.params[1]
            sample_p_value = sample_results.pvalues[1]

            st.session_state["sampling_history"].append(
                {"sample": len(st.session_state["sampling_history"]) + 1, "slope": sample_slope, "p_value": sample_p_value}
            )
            st.session_state["_latest_sample_x"] = sample_x
            st.session_state["_latest_sample_y"] = sample_y

        if st.button("Clear history"):
            st.session_state["sampling_history"] = []
            st.session_state.pop("_latest_sample_x", None)
            st.session_state.pop("_latest_sample_y", None)
            st.rerun()

        if st.session_state["sampling_history"]:
            history_df = pd.DataFrame(st.session_state["sampling_history"])
            significant_count = (history_df["p_value"] < 0.05).sum()
            st.markdown(f"**Samples drawn:** {len(history_df)}")
            st.markdown(f"**Came back \"significant\" (p < 0.05):** {significant_count} / {len(history_df)}")
            st.dataframe(history_df.round(4), use_container_width=True, hide_index=True)
        else:
            st.info("Click **Generate another sample** to draw your first sample.")

    with sampling_plot_col:
        if "_latest_sample_x" in st.session_state:
            latest_x = st.session_state["_latest_sample_x"]
            latest_y = st.session_state["_latest_sample_y"]
            latest_row = st.session_state["sampling_history"][-1]

            sample_figure = go.Figure()
            sample_figure.add_trace(
                go.Scatter(x=latest_x, y=latest_y, mode="markers", marker=dict(size=10, color="#2563eb"), name="sample")
            )
            fit_line_x = np.linspace(0, 10, 50)
            sample_figure.add_trace(
                go.Scatter(
                    x=fit_line_x, y=latest_row["slope"] * fit_line_x + (latest_y.mean() - latest_row["slope"] * latest_x.mean()),
                    mode="lines", line=dict(color="#dc2626", width=3),
                    name=f"fitted slope = {latest_row['slope']:.2f}, p = {latest_row['p_value']:.3f}",
                )
            )
            sample_figure.update_layout(
                title=f"Most Recent Sample (true slope = {true_slope})",
                xaxis_title="x", yaxis_title="y",
                template="plotly_white", height=450,
            )
            st.plotly_chart(sample_figure, use_container_width=True)

        if len(st.session_state["sampling_history"]) >= 3:
            history_df = pd.DataFrame(st.session_state["sampling_history"])
            slope_history_figure = go.Figure()
            slope_history_figure.add_trace(
                go.Scatter(x=history_df["sample"], y=history_df["slope"], mode="markers+lines",
                           marker=dict(size=9, color="#7c3aed"), line=dict(color="#7c3aed", width=1))
            )
            slope_history_figure.add_hline(y=true_slope, line=dict(color="#16a34a", width=2, dash="dash"))
            slope_history_figure.update_layout(
                title="Estimated Slope Across Your Samples (green = true slope)",
                xaxis_title="Sample #", yaxis_title="Estimated slope",
                template="plotly_white", height=350,
            )
            st.plotly_chart(slope_history_figure, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 8 — Logistic Regression
# ---------------------------------------------------------------------------
with tab8:
    st.subheader("Logistic Regression")
    st.write(
        "Ordinary linear regression can't stay inside 0-1 for a pass/fail outcome. Logistic regression "
        "squashes the weighted sum through a sigmoid so the output is always a valid probability."
    )

    logistic_control_col, logistic_plot_col = st.columns([1, 2])

    logistic_coef = logistic_model.coef_[0][0]
    logistic_intercept = logistic_model.intercept_[0]

    with logistic_control_col:
        threshold = st.slider(
            "Decision threshold", 0.0, 1.0, st.session_state["classification_threshold"], 0.05,
            key="classification_threshold",
            help="Predictions with probability >= this value are classified as 'pass'.",
        )

        probabilities = sigmoid(logistic_coef * pass_df["hours_studied"] + logistic_intercept)
        predicted_classes = (probabilities >= threshold).astype(int)
        actual_classes = pass_df["passed"]

        true_positive = int(((predicted_classes == 1) & (actual_classes == 1)).sum())
        true_negative = int(((predicted_classes == 0) & (actual_classes == 0)).sum())
        false_positive = int(((predicted_classes == 1) & (actual_classes == 0)).sum())
        false_negative = int(((predicted_classes == 0) & (actual_classes == 1)).sum())
        accuracy = (true_positive + true_negative) / len(actual_classes)

        st.markdown(f"**Threshold:** {threshold:.2f}")
        st.markdown(f"**Accuracy at this threshold:** {accuracy:.3f}")

        confusion_df = pd.DataFrame(
            {"Predicted: fail": [true_negative, false_negative], "Predicted: pass": [false_positive, true_positive]},
            index=["Actual: fail", "Actual: pass"],
        )
        st.dataframe(confusion_df, use_container_width=True)

        st.markdown("---")
        st.markdown("**Predict for a new student**")
        new_hours_logistic = st.number_input(
            "Hours studied", min_value=0.0, max_value=15.0, value=5.0, step=0.5, key="new_hours_logistic"
        )
        new_probability = sigmoid(logistic_coef * new_hours_logistic + logistic_intercept)
        new_class = "pass" if new_probability >= threshold else "fail"
        st.markdown(f"Predicted probability of passing: **{new_probability:.3f}**")
        st.markdown(f"Predicted class at threshold {threshold:.2f}: **{new_class}**")

    with logistic_plot_col:
        curve_x = np.linspace(0, 10, 200)
        curve_y = sigmoid(logistic_coef * curve_x + logistic_intercept)

        logistic_figure = go.Figure()
        for predicted_value, color, label in [(0, "#dc2626", "predicted: fail"), (1, "#16a34a", "predicted: pass")]:
            subset_mask = predicted_classes == predicted_value
            logistic_figure.add_trace(
                go.Scatter(
                    x=pass_df.loc[subset_mask, "hours_studied"], y=pass_df.loc[subset_mask, "passed"],
                    mode="markers", marker=dict(size=11, color=color), name=label,
                )
            )
        logistic_figure.add_trace(
            go.Scatter(x=curve_x, y=curve_y, mode="lines", line=dict(color="#2563eb", width=3), name="fitted sigmoid")
        )
        logistic_figure.add_hline(y=threshold, line=dict(color="#f59e0b", width=2, dash="dash"))
        logistic_figure.add_trace(
            go.Scatter(
                x=[new_hours_logistic], y=[new_probability], mode="markers",
                marker=dict(size=14, color="#7c3aed", symbol="star"), name="new prediction",
            )
        )
        logistic_figure.update_layout(
            title="Logistic Regression Fit — points colored by current predicted class",
            xaxis_title="Hours studied", yaxis_title="Probability of passing",
            template="plotly_white", height=550,
        )
        st.plotly_chart(logistic_figure, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 9 — Decision Boundary
# ---------------------------------------------------------------------------
with tab9:
    st.subheader("Decision Boundary")
    st.write(
        "With two features, logistic regression separates the feature plane with a straight boundary line "
        "instead of a single threshold point."
    )

    boundary_w1, boundary_w2 = logistic_model_2d.coef_[0]
    boundary_b = logistic_model_2d.intercept_[0]

    boundary_x1 = np.linspace(pass_2d_df["hours_studied"].min(), pass_2d_df["hours_studied"].max(), 50)
    boundary_x2 = -(boundary_b + boundary_w1 * boundary_x1) / boundary_w2

    boundary_figure = go.Figure()
    for passed_value, color, label in [(0, "#dc2626", "failed"), (1, "#16a34a", "passed")]:
        subset = pass_2d_df[pass_2d_df["passed"] == passed_value]
        boundary_figure.add_trace(
            go.Scatter(
                x=subset["hours_studied"], y=subset["practice_problems"], mode="markers",
                marker=dict(size=10, color=color), name=label,
                hovertemplate="Hours: %{x}<br>Problems: %{y}<extra></extra>",
            )
        )
    boundary_figure.add_trace(
        go.Scatter(
            x=boundary_x1, y=boundary_x2, mode="lines",
            line=dict(color="#7c3aed", width=3, dash="dash"), name="decision boundary (p = 0.5)",
        )
    )
    boundary_figure.update_layout(
        title="Decision Boundary: Hours Studied + Practice Problems",
        xaxis_title="Hours studied", yaxis_title="Practice problems",
        template="plotly_white", height=600,
    )
    st.plotly_chart(boundary_figure, use_container_width=True)

    st.markdown(
        f"**Fitted model:** coefficients = ({boundary_w1:.3f}, {boundary_w2:.3f}), intercept = {boundary_b:.3f}  \n"
        f"Everything on one side of the dashed line is predicted \"pass,\" everything on the other \"fail.\""
    )
