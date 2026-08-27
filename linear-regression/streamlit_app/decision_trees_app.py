"""Decision Trees & Random Forest Playground — an interactive companion to the Session 4/5 notebooks.

Run with: streamlit run streamlit_app/decision_trees_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="Decision Trees & Random Forest Playground", layout="wide")


@st.cache_data
def load_classification_3d_data():
    return pd.read_csv(DATA_DIR / "exam_pass_3d.csv")


@st.cache_data
def load_classification_2d_data():
    return pd.read_csv(DATA_DIR / "exam_pass_2d.csv")


pass_3d_df = load_classification_3d_data()
pass_2d_df = load_classification_2d_data()

PASS_3D_TRAIN, PASS_3D_TEST = train_test_split(
    pass_3d_df, test_size=0.3, random_state=0, stratify=pass_3d_df["passed"]
)
PASS_2D_TRAIN, PASS_2D_TEST = train_test_split(
    pass_2d_df, test_size=0.3, random_state=0, stratify=pass_2d_df["passed"]
)

N_ESTIMATORS_SWEEP = [1, 2, 5, 10, 25, 50, 100, 200]
BOUNDARY_FILL_COLORSCALE = [[0, "#fca5a5"], [1, "#86efac"]]


@st.cache_data
def compute_n_estimators_sweep():
    feature_columns = ["hours_studied", "practice_problems"]
    accuracies = []
    for n_estimators in N_ESTIMATORS_SWEEP:
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        forest.fit(PASS_3D_TRAIN[feature_columns], PASS_3D_TRAIN["passed"])
        test_predictions = forest.predict(PASS_3D_TEST[feature_columns])
        accuracies.append(float((test_predictions == PASS_3D_TEST["passed"]).mean()))
    return accuracies


def compute_grid_predictions(classifier, feature_columns, x_range, y_range, resolution=120):
    x_min, x_max = x_range
    y_min, y_max = y_range
    grid_x1, grid_x2 = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid_frame = pd.DataFrame({feature_columns[0]: grid_x1.ravel(), feature_columns[1]: grid_x2.ravel()})
    grid_predictions = classifier.predict(grid_frame).reshape(grid_x1.shape)
    return grid_x1, grid_x2, grid_predictions


def build_boundary_figure(classifier, dataframe, feature_columns, title):
    x_min, x_max = dataframe[feature_columns[0]].min() - 0.5, dataframe[feature_columns[0]].max() + 0.5
    y_min, y_max = dataframe[feature_columns[1]].min() - 1, dataframe[feature_columns[1]].max() + 1
    grid_x1, grid_x2, grid_predictions = compute_grid_predictions(
        classifier, feature_columns, (x_min, x_max), (y_min, y_max)
    )

    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            x=grid_x1[0], y=grid_x2[:, 0], z=grid_predictions,
            colorscale=BOUNDARY_FILL_COLORSCALE, opacity=0.35,
            showscale=False, hoverinfo="skip",
        )
    )
    for passed_value, color, label in [(0, "#dc2626", "actual: failed"), (1, "#16a34a", "actual: passed")]:
        subset = dataframe[dataframe["passed"] == passed_value]
        figure.add_trace(
            go.Scatter(
                x=subset[feature_columns[0]], y=subset[feature_columns[1]], mode="markers",
                marker=dict(size=8, color=color, line=dict(width=1, color="white")), name=label,
                hovertemplate=f"{feature_columns[0]}: %{{x}}<br>{feature_columns[1]}: %{{y}}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title, xaxis_title=feature_columns[0], yaxis_title=feature_columns[1],
        template="plotly_white", height=420,
        xaxis=dict(range=[x_min, x_max]), yaxis=dict(range=[y_min, y_max]),
    )
    return figure


st.session_state.setdefault("tree_max_depth", 3)
st.session_state.setdefault("forest_n_estimators", 50)

st.title("Decision Trees & Random Forest Playground")
st.caption("A companion to Session 4 (decision trees) and Session 5 (random forest).")

tab1, tab2 = st.tabs(["1. Decision Tree — Boundary", "2. Random Forest"])

# ---------------------------------------------------------------------------
# TAB 1 — Decision Tree Boundary
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Decision Tree — Boundary")
    st.write(
        "Session 3's logistic regression drew one straight boundary line. A tree instead asks a "
        "sequence of yes/no questions, carving the feature plane into rectangles."
    )

    control_col, plot_col = st.columns([1, 2])
    feature_columns = ["hours_studied", "practice_problems"]

    with control_col:
        max_depth = st.slider("Tree max depth", 1, 20, st.session_state["tree_max_depth"], 1, key="tree_max_depth")
        use_session3_data = st.checkbox("Use Session 3's original 2-feature dataset instead", value=False)
        show_logistic_boundary = st.checkbox("Overlay logistic regression boundary", value=True)

        if use_session3_data:
            boundary_df, boundary_train, boundary_test = pass_2d_df, PASS_2D_TRAIN, PASS_2D_TEST
        else:
            boundary_df, boundary_train, boundary_test = pass_3d_df, PASS_3D_TRAIN, PASS_3D_TEST

        tree_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        tree_classifier.fit(boundary_train[feature_columns], boundary_train["passed"])

        tree_train_accuracy = tree_classifier.score(boundary_train[feature_columns], boundary_train["passed"])
        tree_test_accuracy = tree_classifier.score(boundary_test[feature_columns], boundary_test["passed"])
        st.markdown(f"**Tree accuracy:** train = {tree_train_accuracy:.3f}, test = {tree_test_accuracy:.3f}")

        if show_logistic_boundary:
            logistic_boundary_model = LogisticRegression()
            logistic_boundary_model.fit(boundary_train[feature_columns], boundary_train["passed"])
            logistic_test_accuracy = logistic_boundary_model.score(
                boundary_test[feature_columns], boundary_test["passed"]
            )
            st.markdown(f"**Logistic regression test accuracy:** {logistic_test_accuracy:.3f}")

    with plot_col:
        boundary_figure = build_boundary_figure(
            tree_classifier, boundary_df, feature_columns,
            f"Decision Tree Boundary (depth={max_depth}) — filled region is the tree's prediction",
        )

        if show_logistic_boundary:
            x_min, x_max = boundary_df["hours_studied"].min() - 0.5, boundary_df["hours_studied"].max() + 0.5
            boundary_w1, boundary_w2 = logistic_boundary_model.coef_[0]
            boundary_b = logistic_boundary_model.intercept_[0]
            line_x1 = np.linspace(x_min, x_max, 50)
            line_x2 = -(boundary_b + boundary_w1 * line_x1) / boundary_w2
            boundary_figure.add_trace(
                go.Scatter(
                    x=line_x1, y=line_x2, mode="lines",
                    line=dict(color="#7c3aed", width=3, dash="dash"), name="logistic regression boundary",
                )
            )

        st.plotly_chart(boundary_figure, use_container_width=True)
        st.caption("The filled region is the tree's predicted class; points are colored by their real outcome.")

# ---------------------------------------------------------------------------
# TAB 2 — Random Forest
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Random Forest")
    st.write(
        "One deep tree can memorize noise. A forest averages the votes of many trees, each trained on "
        "a different bootstrap sample — smoother predictions, less overfitting."
    )

    control_col, plot_col = st.columns([1, 2])
    feature_columns_2d = ["hours_studied", "practice_problems"]
    feature_columns_3d = ["hours_studied", "practice_problems", "sleep_hours"]
    IMPORTANCE_TREE_DEPTH = 3

    with control_col:
        n_estimators = st.slider(
            "Number of trees (n_estimators)", 1, 200, st.session_state["forest_n_estimators"], 1,
            key="forest_n_estimators",
        )
        st.caption(
            "Compared against a single unconstrained tree (max_depth=None) — the kind that memorizes "
            "its training data instead of learning the general rule."
        )

        forest_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        forest_classifier.fit(PASS_3D_TRAIN[feature_columns_2d], PASS_3D_TRAIN["passed"])
        forest_train_accuracy = forest_classifier.score(PASS_3D_TRAIN[feature_columns_2d], PASS_3D_TRAIN["passed"])
        forest_test_accuracy = forest_classifier.score(PASS_3D_TEST[feature_columns_2d], PASS_3D_TEST["passed"])
        st.markdown(f"**Forest accuracy:** train = {forest_train_accuracy:.3f}, test = {forest_test_accuracy:.3f}")

        overfit_tree = DecisionTreeClassifier(max_depth=None, random_state=0)
        overfit_tree.fit(PASS_3D_TRAIN[feature_columns_2d], PASS_3D_TRAIN["passed"])
        overfit_tree_train_accuracy = overfit_tree.score(PASS_3D_TRAIN[feature_columns_2d], PASS_3D_TRAIN["passed"])
        overfit_tree_test_accuracy = overfit_tree.score(PASS_3D_TEST[feature_columns_2d], PASS_3D_TEST["passed"])
        st.markdown(
            f"**Single overfit tree accuracy:** train = {overfit_tree_train_accuracy:.3f}, "
            f"test = {overfit_tree_test_accuracy:.3f}"
        )

    with plot_col:
        boundary_col1, boundary_col2 = st.columns(2)
        with boundary_col1:
            st.plotly_chart(
                build_boundary_figure(overfit_tree, pass_3d_df, feature_columns_2d, "Single Tree (max_depth=None)"),
                use_container_width=True,
            )
        with boundary_col2:
            st.plotly_chart(
                build_boundary_figure(forest_classifier, pass_3d_df, feature_columns_2d, "Random Forest"),
                use_container_width=True,
            )
        st.caption(
            "The single tree's boundary is jagged — tiny rectangles carved out to fit individual noisy "
            "points. The forest's boundary is smoother and generalizes better, even though both were "
            "trained on the exact same data."
        )

        single_tree_for_importance = DecisionTreeClassifier(max_depth=IMPORTANCE_TREE_DEPTH, random_state=0)
        single_tree_for_importance.fit(PASS_3D_TRAIN[feature_columns_3d], PASS_3D_TRAIN["passed"])
        forest_for_importance = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        forest_for_importance.fit(PASS_3D_TRAIN[feature_columns_3d], PASS_3D_TRAIN["passed"])

        importance_figure = go.Figure()
        importance_figure.add_trace(
            go.Bar(
                x=feature_columns_3d, y=single_tree_for_importance.feature_importances_,
                name="single tree", marker_color="#dc2626",
            )
        )
        importance_figure.add_trace(
            go.Bar(
                x=feature_columns_3d, y=forest_for_importance.feature_importances_,
                name="random forest", marker_color="#16a34a",
            )
        )
        importance_figure.update_layout(
            title="Feature Importance: Single Tree vs Random Forest",
            xaxis_title="Feature", yaxis_title="Importance",
            barmode="group", template="plotly_white", height=350,
        )
        st.plotly_chart(importance_figure, use_container_width=True)
        st.caption(
            "sleep_hours is pure noise. A single tree tends to ignore it almost completely; the forest "
            "gives it a small-but-nonzero score because each split only considers a random subset of features."
        )

        sweep_accuracies = compute_n_estimators_sweep()
        sweep_figure = go.Figure()
        sweep_figure.add_trace(
            go.Scatter(
                x=N_ESTIMATORS_SWEEP, y=sweep_accuracies, mode="markers+lines",
                marker=dict(size=9, color="#7c3aed"), line=dict(color="#7c3aed", width=2), name="sweep",
            )
        )
        sweep_figure.add_trace(
            go.Scatter(
                x=[n_estimators], y=[forest_test_accuracy], mode="markers",
                marker=dict(size=14, color="#f59e0b", symbol="star"), name="current setting",
            )
        )
        sweep_figure.update_layout(
            title="Test Accuracy vs Number of Trees",
            xaxis_title="n_estimators", yaxis_title="Test accuracy",
            template="plotly_white", height=350,
        )
        st.plotly_chart(sweep_figure, use_container_width=True)
