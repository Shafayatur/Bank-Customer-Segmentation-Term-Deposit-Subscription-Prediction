import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import base64
import io
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Bank Customer Analytics",
)

server = app.server


# ─────────────────────────────────────────────────────────────────────────────
# Colors and chart layout
# ─────────────────────────────────────────────────────────────────────────────

BLUE = "#1a6ef5"
GREEN = "#2e7d51"
AMBER = "#d4831a"
RED = "#c0392b"
DARK = "#0f1923"
LIGHT = "#f4f6f8"
BORDER = "#e8ecf0"

PLOT_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font_family="DM Sans",
    font_color=DARK,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#f0f2f5", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="DM Sans"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def pct_yes(series):
    return (series == "yes").mean() * 100


def sub_rate_by(df, col):
    return (
        df.groupby(col)["y"]
        .apply(pct_yes)
        .reset_index(name="subscription_rate")
        .sort_values("subscription_rate", ascending=False)
    )


def make_bar(df_plot, x, y, title="", horizontal=False, color=BLUE, text_suffix="%"):
    orientation = "h" if horizontal else "v"

    fig = go.Figure(
        go.Bar(
            x=df_plot[y] if horizontal else df_plot[x],
            y=df_plot[x] if horizontal else df_plot[y],
            orientation=orientation,
            marker_color=color,
            marker_line_width=0,
            text=[f"{v:.1f}{text_suffix}" for v in df_plot[y]],
            textposition="outside",
        )
    )

    layout = {**PLOT_LAYOUT, "title": dict(text=title, font_size=13)}

    if horizontal:
        layout["xaxis"] = dict(showgrid=True, gridcolor="#f0f2f5", zeroline=False)
        layout["yaxis"] = dict(showgrid=False, zeroline=False)

    fig.update_layout(**layout)
    return fig


def card(children, style=None):
    base = {
        "background": "white",
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "1.25rem 1.5rem",
        "boxShadow": "0 1px 4px rgba(0,0,0,.05)",
        "height": "100%",
    }

    if style:
        base.update(style)

    return html.Div(children, style=base)


def section_header(text):
    return html.Div(
        text,
        style={
            "fontSize": "13px",
            "fontWeight": "600",
            "color": DARK,
            "borderLeft": f"3px solid {BLUE}",
            "paddingLeft": "10px",
            "marginBottom": "14px",
        },
    )


def insight(text):
    return html.Div(
        dcc.Markdown(text),
        style={
            "background": "#f0f5ff",
            "borderLeft": f"3px solid {BLUE}",
            "borderRadius": "0 8px 8px 0",
            "padding": ".7rem 1rem",
            "fontSize": "13px",
            "color": "#2c3e50",
            "lineHeight": "1.6",
            "marginTop": "12px",
        },
    )


def metric_card(label, value, delta="", delta_neg=False):
    return card(
        [
            html.Div(
                label,
                style={
                    "fontSize": "11px",
                    "color": "#7a8694",
                    "fontWeight": "500",
                    "letterSpacing": ".05em",
                    "textTransform": "uppercase",
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "28px",
                    "fontWeight": "600",
                    "color": DARK,
                    "lineHeight": "1.1",
                },
            ),
            html.Div(
                delta,
                style={
                    "fontSize": "12px",
                    "marginTop": "4px",
                    "color": RED if delta_neg else GREEN,
                },
            )
            if delta
            else None,
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sample data
# This is only for demo when no CSV is uploaded.
# For matching notebook output, upload the same CSV used in the notebook.
# ─────────────────────────────────────────────────────────────────────────────

def get_sample_data():
    np.random.seed(42)
    n = 3000

    jobs = [
        "admin.",
        "blue-collar",
        "technician",
        "services",
        "management",
        "retired",
        "self-employed",
        "entrepreneur",
        "housemaid",
        "student",
    ]

    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 88, n),
            "job": np.random.choice(
                jobs, n, p=[.25, .22, .16, .09, .07, .06, .05, .04, .03, .03]
            ),
            "marital": np.random.choice(
                ["married", "single", "divorced"], n, p=[.60, .28, .12]
            ),
            "education": np.random.choice(
                [
                    "university.degree",
                    "high.school",
                    "basic.9y",
                    "professional.course",
                    "basic.6y",
                ],
                n,
                p=[.30, .23, .15, .13, .19],
            ),
            "default": np.random.choice(["no", "yes"], n, p=[.97, .03]),
            "housing": np.random.choice(["no", "yes"], n, p=[.45, .55]),
            "loan": np.random.choice(["no", "yes"], n, p=[.82, .18]),
            "contact": np.random.choice(["cellular", "telephone"], n, p=[.64, .36]),
            "month": np.random.choice(
                [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ],
                n,
            ),
            "campaign": np.clip(np.random.poisson(2.5, n), 1, 20),
            "pdays": np.where(
                np.random.rand(n) < .15, np.random.randint(1, 30, n), -1
            ),
            "previous": np.random.poisson(0.4, n),
            "poutcome": np.random.choice(
                ["nonexistent", "failure", "success"], n, p=[.86, .10, .04]
            ),
            "emp.var.rate": np.random.uniform(-3.5, 1.5, n).round(1),
            "cons.price.idx": np.random.uniform(92, 95, n).round(3),
            "cons.conf.idx": np.random.uniform(-51, -26, n).round(1),
            "euribor3m": np.random.uniform(0.6, 5.1, n).round(3),
            "nr.employed": np.random.choice([4963.6, 5099.1, 5228.1, 5195.8], n),
            "balance": np.random.normal(1500, 3000, n).round(0),
            "y": np.random.choice(["no", "yes"], n, p=[.889, .111]),
        }
    )

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 40, 60, 100],
        labels=["Young", "Adult", "Middle Age", "Senior"],
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# This is aligned with notebook style:
# unknown -> NaN
# pdays 999 -> -1
# duration removed
# age_group created and KEPT for modelling
# ─────────────────────────────────────────────────────────────────────────────

def process_df(df):
    df = df.copy()

    df = df.replace("unknown", np.nan)

    if "pdays" in df.columns:
        df["pdays"] = df["pdays"].replace(999, -1)

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 40, 60, 100],
            labels=["Young", "Adult", "Middle Age", "Senior"],
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# This version is made to match the notebook pipeline:
# 1. keep age_group
# 2. pd.get_dummies(drop_first=True)
# 3. target = y_yes
# 4. same train_test_split
# 5. same model parameters
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(df):
    df_m = df.copy()

    if "y" not in df_m.columns:
        raise ValueError("Target column 'y' was not found in the dataset.")

    # Notebook-style one-hot encoding
    df_m = pd.get_dummies(df_m, drop_first=True)

    if "y_yes" not in df_m.columns:
        raise ValueError(
            "After one-hot encoding, target column 'y_yes' was not found. "
            "Make sure the original target column 'y' contains values 'yes' and 'no'."
        )

    X = df_m.drop("y_yes", axis=1)
    y = df_m["y_yes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(Xtr_sc, y_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    dt = DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
    )
    dt.fit(X_train, y_train)

    results = {}

    model_specs = [
        ("Logistic Regression", lr, Xte_sc),
        ("Random Forest", rf, X_test),
        ("Decision Tree", dt, X_test),
    ]

    for name, model, Xte in model_specs:
        preds = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]

        # Convert target and predictions to integer labels.
        # This avoids the Model Error: '1' problem when y_yes becomes True/False.
        y_test_int = y_test.astype(int)
        preds_int = pd.Series(preds).astype(int)

        rep = classification_report(
            y_test_int,
            preds_int,
            labels=[0, 1],
            target_names=["0", "1"],
            output_dict=True,
            zero_division=0,
        )

        results[name] = {
            "accuracy": accuracy_score(y_test_int, preds_int),
            "auc": roc_auc_score(y_test_int, proba),
            "recall_yes": rep["1"]["recall"],
            "precision_yes": rep["1"]["precision"],
            "f1_yes": rep["1"]["f1-score"],
            "cm": confusion_matrix(y_test_int, preds_int, labels=[0, 1]).tolist(),
            "roc": roc_curve(y_test_int, proba),
            "model": model,
            "feature_names": list(X.columns),
        }

    return results, y_test, X_train, X_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# K-Means
# Elbow range changed to 1-10 to match notebook.
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(df):
    cluster_cols = [
        c
        for c in [
            "age",
            "campaign",
            "previous",
            "pdays",
            "euribor3m",
            "emp.var.rate",
            "nr.employed",
        ]
        if c in df.columns
    ]

    if not cluster_cols:
        raise ValueError("No valid numerical columns found for clustering.")

    X_c = df[cluster_cols].copy()
    X_c = X_c.fillna(X_c.median(numeric_only=True))

    scaler_c = StandardScaler()
    X_sc = scaler_c.fit_transform(X_c)

    wcss = [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc).inertia_
        for k in range(1, 11)
    ]

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)

    return labels, wcss, cluster_cols


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

STYLES = """
* { box-sizing: border-box; }
body { font-family: 'DM Sans', sans-serif; background: #f4f6f8; margin: 0; }

.sidebar {
    width: 220px; min-height: 100vh; background: #0f1923;
    padding: 1.5rem 1.25rem; position: fixed; top: 0; left: 0;
    border-right: 1px solid #1e2d3d; z-index: 100;
}

.sidebar-title { color: white; font-size: 16px; font-weight: 600; margin-bottom: 4px; }
.sidebar-sub   { color: #8a9bb0; font-size: 11px; margin-bottom: 1.5rem; white-space: pre-line; }

.nav-label {
    color: #5a7a99; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px;
}

.nav-btn {
    display: block; width: 100%; text-align: left; background: transparent;
    border: none; color: #c9d4df; padding: .5rem .75rem; border-radius: 7px;
    font-size: 13px; font-family: 'DM Sans', sans-serif; cursor: pointer;
    margin-bottom: 3px; transition: background .15s;
}

.nav-btn:hover  { background: #1e2d3d; color: white; }
.nav-btn.active { background: #1a6ef5; color: white; font-weight: 500; }

.main-content { margin-left: 220px; padding: 2rem 2.5rem; min-height: 100vh; }

.page-title {
    font-size: 22px; font-weight: 600; color: #0f1923; margin-bottom: 1.5rem;
}

.upload-area {
    border: 2px dashed #2d4a6b; border-radius: 10px; padding: 1rem;
    text-align: center; color: #8a9bb0; font-size: 12px; margin-top: 1.5rem;
    background: #131f2b;
}

.data-badge {
    background: #1e2d3d; border-radius: 6px; padding: 4px 10px;
    font-size: 11px; color: #8a9bb0; margin-top: 1rem; text-align: center;
}

.seg-card {
    border-radius: 12px; padding: 1.1rem 1.25rem;
    font-family: 'DM Sans', sans-serif;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────

PAGES = ["Overview", "Demographics", "Financial", "Campaign", "ML Models", "Segments"]
PAGE_ICONS = ["📊", "👥", "💰", "📣", "🤖", "🔵"]

app.layout = html.Div(
    [
        

        dcc.Store(id="store-data"),
        dcc.Store(id="store-page", data="Overview"),

        html.Div(
            [
                html.Div("🏦 Bank Analytics", className="sidebar-title"),
                html.Div(
                    "Customer Segmentation\n& Term Deposit Prediction",
                    className="sidebar-sub",
                ),

                html.Div("Dataset", className="nav-label"),

                dcc.Upload(
                    id="upload-data",
                    children=html.Div(
                        [
                            "📁 Drop CSV here",
                            html.Br(),
                            html.Span(
                                "or click to browse",
                                style={"fontSize": "11px"},
                            ),
                        ]
                    ),
                    className="upload-area",
                    multiple=False,
                ),

                html.Div(
                    id="data-badge",
                    className="data-badge",
                    children="Sample data loaded",
                ),

                html.Hr(style={"borderColor": "#1e2d3d", "margin": "1.25rem 0"}),

                html.Div("Navigation", className="nav-label"),

                html.Div(
                    [
                        html.Button(
                            f"{icon} {name}",
                            id=f"nav-{name}",
                            n_clicks=0,
                            className="nav-btn active" if name == "Overview" else "nav-btn",
                        )
                        for icon, name in zip(PAGE_ICONS, PAGES)
                    ]
                ),

                html.Hr(style={"borderColor": "#1e2d3d", "margin": "1.25rem 0"}),

                html.Div(
                    "v1.0 · Dash by Plotly",
                    style={"color": "#3a5a7a", "fontSize": "11px"},
                ),
            ],
            className="sidebar",
        ),

        html.Div(
            [
                html.Div(id="page-content"),
            ],
            className="main-content",
        ),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Upload callback
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-data", "data"),
    Output("data-badge", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def upload_data(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=None, engine="python")
    df = process_df(df)

    return df.to_json(date_format="iso", orient="split"), f"✅ {filename} · {len(df):,} rows"


# ─────────────────────────────────────────────────────────────────────────────
# Navigation callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-page", "data"),
    [Input(f"nav-{name}", "n_clicks") for name in PAGES],
    prevent_initial_call=True,
)
def switch_page(*args):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "Overview"

    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return btn_id.replace("nav-", "")


@app.callback(
    [Output(f"nav-{name}", "className") for name in PAGES],
    Input("store-page", "data"),
)
def update_nav(page):
    return ["nav-btn active" if name == page else "nav-btn" for name in PAGES]


# ─────────────────────────────────────────────────────────────────────────────
# Page renderer
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("page-content", "children"),
    Input("store-page", "data"),
    Input("store-data", "data"),
)
def render_page(page, json_data):
    if json_data:
        df = pd.read_json(io.StringIO(json_data), orient="split")

        if "age" in df.columns and "age_group" not in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 40, 60, 100],
                labels=["Young", "Adult", "Middle Age", "Senior"],
            )
    else:
        df = get_sample_data()

    if page == "Overview":
        return page_overview(df)
    if page == "Demographics":
        return page_demographics(df)
    if page == "Financial":
        return page_financial(df)
    if page == "Campaign":
        return page_campaign(df)
    if page == "ML Models":
        return page_models(df)
    if page == "Segments":
        return page_segments(df)

    return html.Div("Page not found")


# ══════════════════════════════════════════════════════════════════════════════
# Page builders
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(df):
    total = len(df)
    sub_n = (df["y"] == "yes").sum()
    sub_pct = sub_n / total * 100
    avg_age = df["age"].mean()

    if "contact" in df.columns and len(df[df["contact"] == "cellular"]) > 0:
        cell_r = pct_yes(df[df["contact"] == "cellular"]["y"])
    else:
        cell_r = 0

    vc = df["y"].value_counts()

    fig_pie = go.Figure(
        go.Pie(
            labels=["Not Subscribed", "Subscribed"],
            values=[vc.get("no", 0), vc.get("yes", 0)],
            hole=.62,
            marker_colors=["#e8ecf0", BLUE],
            textinfo="label+percent",
            textfont_size=12,
        )
    )

    fig_pie.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white",
        font_family="DM Sans",
        height=260,
    )

    fig_pie.add_annotation(
        text=f"<b>{sub_pct:.1f}%</b><br>subscribed",
        x=.5,
        y=.5,
        showarrow=False,
        font_size=14,
        font_color=DARK,
        align="center",
    )

    fig_job = make_bar(
        sub_rate_by(df, "job"),
        "job",
        "subscription_rate",
        "Subscription Rate by Job",
        horizontal=True,
    )

    month_order = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    if "month" in df.columns:
        md = sub_rate_by(df, "month")
        md["month"] = pd.Categorical(md["month"], categories=month_order, ordered=True)
        md = md.sort_values("month")

        fig_month = px.line(
            md,
            x="month",
            y="subscription_rate",
            markers=True,
            color_discrete_sequence=[BLUE],
            labels={"month": "", "subscription_rate": "Rate (%)"},
            title="Monthly Subscription Rate",
        )

        fig_month.update_traces(line_width=2.5, marker_size=7)
        fig_month.update_layout(**PLOT_LAYOUT)
    else:
        fig_month = go.Figure()

    return html.Div(
        [
            html.Div("Overview", className="page-title"),

            dbc.Row(
                [
                    dbc.Col(metric_card("Total Customers", f"{total:,}", "Full dataset"), md=3),
                    dbc.Col(
                        metric_card(
                            "Subscribed",
                            f"{sub_n:,}",
                            f"{sub_pct:.1f}% conversion rate",
                        ),
                        md=3,
                    ),
                    dbc.Col(metric_card("Avg Age", f"{avg_age:.0f}", "years"), md=3),
                    dbc.Col(
                        metric_card(
                            "Cellular Conversion",
                            f"{cell_r:.1f}%",
                            "vs telephone",
                        ),
                        md=3,
                    ),
                ],
                className="mb-4",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                section_header("Subscription Split"),
                                dcc.Graph(
                                    figure=fig_pie,
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        card(
                            [
                                section_header("By Job Type"),
                                dcc.Graph(
                                    figure=fig_job,
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        md=8,
                    ),
                ],
                className="mb-3",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                section_header("Monthly Trend"),
                                dcc.Graph(
                                    figure=fig_month,
                                    config={"displayModeBar": False},
                                ),
                                insight(
                                    "**Mar, Sep, Oct, Dec** usually show stronger subscription rates. "
                                    "Use the uploaded dataset to confirm the real business trend."
                                ),
                            ]
                        ),
                        md=12,
                    ),
                ]
            ),
        ]
    )


def page_demographics(df):
    fig_age = (
        make_bar(
            sub_rate_by(df, "age_group"),
            "age_group",
            "subscription_rate",
            "By Age Group",
        )
        if "age_group" in df.columns
        else go.Figure()
    )

    fig_mar = (
        make_bar(
            sub_rate_by(df, "marital"),
            "marital",
            "subscription_rate",
            "By Marital Status",
            color=AMBER,
        )
        if "marital" in df.columns
        else go.Figure()
    )

    fig_edu = (
        make_bar(
            sub_rate_by(df, "education"),
            "education",
            "subscription_rate",
            "By Education",
            horizontal=True,
            color=GREEN,
        )
        if "education" in df.columns
        else go.Figure()
    )

    fig_hist = px.histogram(
        df,
        x="age",
        color="y",
        color_discrete_map={"no": "#e8ecf0", "yes": BLUE},
        barmode="overlay",
        nbins=40,
        labels={"age": "Age", "y": "Subscribed"},
        title="Age Distribution",
    )

    fig_hist.update_layout(**PLOT_LAYOUT)

    return html.Div(
        [
            html.Div("Demographics", className="page-title"),

            dbc.Row(
                [
                    dbc.Col(
                        card([dcc.Graph(figure=fig_age, config={"displayModeBar": False})]),
                        md=6,
                    ),
                    dbc.Col(
                        card([dcc.Graph(figure=fig_hist, config={"displayModeBar": False})]),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        card([dcc.Graph(figure=fig_mar, config={"displayModeBar": False})]),
                        md=5,
                    ),
                    dbc.Col(
                        card([dcc.Graph(figure=fig_edu, config={"displayModeBar": False})]),
                        md=7,
                    ),
                ],
                className="mb-3",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        insight(
                            "**Senior and young customer groups** often show different conversion behavior. "
                            "Use this page to compare age, marital status, and education patterns."
                        )
                    )
                ]
            ),
        ]
    )


def page_financial(df):
    fig_housing = (
        make_bar(
            sub_rate_by(df, "housing"),
            "housing",
            "subscription_rate",
            "Housing Loan",
            color=BLUE,
        )
        if "housing" in df.columns
        else go.Figure()
    )

    fig_loan = (
        make_bar(
            sub_rate_by(df, "loan"),
            "loan",
            "subscription_rate",
            "Personal Loan",
            color=AMBER,
        )
        if "loan" in df.columns
        else go.Figure()
    )

    rows = [
        dbc.Row(
            [
                dbc.Col(
                    card([dcc.Graph(figure=fig_housing, config={"displayModeBar": False})]),
                    md=6,
                ),
                dbc.Col(
                    card([dcc.Graph(figure=fig_loan, config={"displayModeBar": False})]),
                    md=6,
                ),
            ],
            className="mb-3",
        )
    ]

    if "balance" in df.columns:
        df_b = df[df["balance"].between(-5000, 20000)]

        fig_box = px.box(
            df_b,
            x="y",
            y="balance",
            color="y",
            color_discrete_map={"no": "#e8ecf0", "yes": BLUE},
            points=False,
            title="Balance Distribution by Subscription",
            labels={"y": "Subscribed", "balance": "Balance"},
        )

        fig_box.update_layout(**PLOT_LAYOUT)

        sub_med = df[df["y"] == "yes"]["balance"].median()
        no_med = df[df["y"] == "no"]["balance"].median()

        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_box,
                                    config={"displayModeBar": False},
                                ),
                                insight(
                                    f"Subscribers have median balance **{sub_med:,.0f}** "
                                    f"vs non-subscribers **{no_med:,.0f}**."
                                ),
                            ]
                        ),
                        md=12,
                    )
                ],
                className="mb-3",
            )
        )

    econ_cols = [
        c for c in ["euribor3m", "emp.var.rate", "cons.conf.idx"] if c in df.columns
    ]

    if econ_cols:
        violin_cols = []

        for ec in econ_cols:
            fig_v = px.violin(
                df,
                x="y",
                y=ec,
                color="y",
                color_discrete_map={"no": "#e8ecf0", "yes": BLUE},
                box=True,
                points=False,
                title=ec,
                labels={"y": ""},
            )

            fig_v.update_layout(
                **{
                    **PLOT_LAYOUT,
                    "showlegend": False,
                    "margin": dict(l=10, r=10, t=40, b=10),
                }
            )

            violin_cols.append(
                dbc.Col(
                    card([dcc.Graph(figure=fig_v, config={"displayModeBar": False})]),
                    md=4,
                )
            )

        rows.append(
            html.Div(
                [
                    html.Div(
                        "Economic Indicators vs Subscription",
                        className="page-title",
                        style={
                            "fontSize": "15px",
                            "marginBottom": "12px",
                            "marginTop": "8px",
                        },
                    ),
                    dbc.Row(violin_cols, className="mb-3"),
                ]
            )
        )

    return html.Div([html.Div("Financial Behavior", className="page-title")] + rows)


def page_campaign(df):
    fig_contact = (
        make_bar(
            sub_rate_by(df, "contact"),
            "contact",
            "subscription_rate",
            "Contact Method",
            color=BLUE,
        )
        if "contact" in df.columns
        else go.Figure()
    )

    fig_poutcome = (
        make_bar(
            sub_rate_by(df, "poutcome"),
            "poutcome",
            "subscription_rate",
            "Previous Outcome",
            color=GREEN,
        )
        if "poutcome" in df.columns
        else go.Figure()
    )

    camp_rows = []

    if "campaign" in df.columns:
        df_c = df.copy()

        df_c["campaign_bucket"] = pd.cut(
            df_c["campaign"],
            bins=[0, 1, 2, 3, 5, 100],
            labels=["1", "2", "3", "4–5", "6+"],
        )

        cb = (
            df_c.groupby("campaign_bucket")["y"]
            .apply(pct_yes)
            .reset_index(name="subscription_rate")
        )

        fig_camp = make_bar(
            cb,
            "campaign_bucket",
            "subscription_rate",
            "Campaign Frequency vs Subscription Rate",
        )

        camp_rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_camp,
                                    config={"displayModeBar": False},
                                ),
                                insight(
                                    "Compare conversion rate by number of campaign contacts. "
                                    "High contact frequency may reduce efficiency."
                                ),
                            ]
                        ),
                        md=12,
                    )
                ],
                className="mb-3",
            )
        )

    if "contact" in df.columns:
        cell_r = pct_yes(df[df["contact"] == "cellular"]["y"])
        tel_r = pct_yes(df[df["contact"] == "telephone"]["y"])
    else:
        cell_r = 0
        tel_r = 0

    if "poutcome" in df.columns:
        suc_r = pct_yes(df[df["poutcome"] == "success"]["y"])
    else:
        suc_r = 0

    return html.Div(
        [
            html.Div("Campaign Analysis", className="page-title"),

            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_contact,
                                    config={"displayModeBar": False},
                                ),
                                insight(
                                    f"Cellular conversion: **{cell_r:.1f}%**. "
                                    f"Telephone conversion: **{tel_r:.1f}%**."
                                ),
                            ]
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_poutcome,
                                    config={"displayModeBar": False},
                                ),
                                insight(
                                    f"Previous campaign success conversion: **{suc_r:.1f}%**."
                                ),
                            ]
                        ),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),

            *camp_rows,
        ]
    )


def page_models(df):
    try:
        results, y_test, X_train, X_test, scaler = train_all_models(df)
    except Exception as e:
        return html.Div(
            [
                html.Div("ML Models", className="page-title"),
                card(
                    [
                        section_header("Model Error"),
                        html.Div(
                            str(e),
                            style={
                                "color": RED,
                                "fontSize": "14px",
                                "lineHeight": "1.6",
                            },
                        ),
                    ]
                ),
            ]
        )

    comp = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": f"{v['accuracy']:.3f}",
                "ROC-AUC": f"{v['auc']:.3f}",
                "Recall (Yes)": f"{v['recall_yes']:.3f}",
                "Precision (Yes)": f"{v['precision_yes']:.3f}",
                "F1 (Yes)": f"{v['f1_yes']:.3f}",
            }
            for name, v in results.items()
        ]
    )

    max_auc = comp["ROC-AUC"].astype(float).max()

    fig_roc = go.Figure()

    for (name, v), color in zip(results.items(), [BLUE, GREEN, AMBER]):
        fpr, tpr, _ = v["roc"]

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                name=f"{name} (AUC={v['auc']:.3f})",
                line=dict(color=color, width=2),
            )
        )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(dash="dash", color="#ccc"),
            showlegend=False,
        )
    )

    fig_roc.update_layout(
        **{
            **PLOT_LAYOUT,
            "title": "ROC Curves",
            "xaxis": dict(
                title="False Positive Rate",
                showgrid=True,
                gridcolor="#f0f2f5",
            ),
            "yaxis": dict(
                title="True Positive Rate",
                showgrid=True,
                gridcolor="#f0f2f5",
            ),
        }
    )

    rf_v = results["Random Forest"]

    # Notebook shows top 15 feature importances.
    fi = (
        pd.Series(
            rf_v["model"].feature_importances_,
            index=rf_v["feature_names"],
        )
        .nlargest(15)
        .sort_values()
    )

    fig_fi = go.Figure(
        go.Bar(
            x=fi.values,
            y=fi.index,
            orientation="h",
            marker_color=BLUE,
            marker_line_width=0,
        )
    )

    fig_fi.update_layout(
        **{
            **PLOT_LAYOUT,
            "title": "Top 15 Feature Importance (Random Forest)",
            "xaxis": dict(showgrid=True, gridcolor="#f0f2f5"),
            "yaxis": dict(showgrid=False),
        }
    )

    cm_figs = []

    for name, v in results.items():
        cm = np.array(v["cm"])

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=[[0, "#f4f6f8"], [1, BLUE]],
            x=["No", "Yes"],
            y=["No", "Yes"],
            title=name,
        )

        fig_cm.update_layout(
            font_family="DM Sans",
            margin=dict(l=0, r=0, t=40, b=0),
            height=220,
            coloraxis_showscale=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        cm_figs.append(
            dbc.Col(
                card([dcc.Graph(figure=fig_cm, config={"displayModeBar": False})]),
                md=4,
            )
        )

    return html.Div(
        [
            html.Div("ML Models", className="page-title"),

            card(
                [
                    section_header("Model Comparison"),

                    dash_table.DataTable(
                        data=comp.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in comp.columns],
                        style_table={
                            "borderRadius": "8px",
                            "overflow": "hidden",
                        },
                        style_header={
                            "backgroundColor": LIGHT,
                            "fontWeight": "600",
                            "fontSize": "12px",
                            "color": DARK,
                            "border": f"1px solid {BORDER}",
                        },
                        style_cell={
                            "fontFamily": "DM Sans",
                            "fontSize": "13px",
                            "padding": "10px 14px",
                            "border": f"1px solid {BORDER}",
                            "textAlign": "center",
                        },
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": f"{{ROC-AUC}} = {max_auc:.3f}"
                                },
                                "backgroundColor": "#f0f5ff",
                                "fontWeight": "600",
                            }
                        ],
                    ),

                    insight(
                        "**Primary metric: ROC-AUC.** Accuracy can be misleading because the dataset is imbalanced. "
                        "This Dash version now follows the notebook-style model pipeline."
                    ),
                ],
                style={"marginBottom": "16px"},
            ),

            dbc.Row(
                [
                    dbc.Col(
                        card([dcc.Graph(figure=fig_roc, config={"displayModeBar": False})]),
                        md=6,
                    ),
                    dbc.Col(
                        card([dcc.Graph(figure=fig_fi, config={"displayModeBar": False})]),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),

            html.Div(
                "Confusion Matrices",
                style={
                    "fontSize": "13px",
                    "fontWeight": "600",
                    "color": DARK,
                    "marginBottom": "12px",
                },
            ),

            dbc.Row(cm_figs, className="mb-3"),
        ]
    )


def page_segments(df):
    try:
        labels, wcss, cluster_cols = run_kmeans(df)
    except Exception as e:
        return html.Div(
            [
                html.Div("Customer Segments", className="page-title"),
                card(
                    [
                        section_header("Clustering Error"),
                        html.Div(
                            str(e),
                            style={
                                "color": RED,
                                "fontSize": "14px",
                                "lineHeight": "1.6",
                            },
                        ),
                    ]
                ),
            ]
        )

    df_cl = df.copy()
    df_cl["cluster"] = labels.astype(str)

    # Notebook-style elbow range: 1 to 10
    fig_elbow = px.line(
        x=range(1, 11),
        y=wcss,
        markers=True,
        labels={"x": "Number of Clusters", "y": "WCSS"},
        color_discrete_sequence=[BLUE],
        title="Elbow Method",
    )

    fig_elbow.add_vline(
        x=3,
        line_dash="dash",
        line_color=RED,
        annotation_text="k=3 selected",
        annotation_position="top right",
    )

    fig_elbow.update_layout(**PLOT_LAYOUT)

    fig_sc = px.scatter(
        df_cl,
        x="age",
        y="campaign",
        color="cluster",
        color_discrete_sequence=[BLUE, GREEN, AMBER],
        opacity=.5,
        title="Cluster Scatter (Age vs Contacts)",
        labels={"campaign": "Campaign Contacts"},
    )

    fig_sc.update_layout(**PLOT_LAYOUT)

    profile = df_cl.groupby("cluster")[cluster_cols].mean().round(2).reset_index()

    if "y" in df_cl.columns:
        profile["subscription_rate_%"] = (
            df_cl.groupby("cluster")["y"].apply(pct_yes).round(1).values
        )

    if "y" in df_cl.columns:
        sub_cl = (
            df_cl.groupby("cluster")["y"]
            .apply(pct_yes)
            .reset_index(name="rate")
        )

        fig_clb = go.Figure(
            go.Bar(
                x=sub_cl["cluster"],
                y=sub_cl["rate"],
                marker_color=[BLUE, GREEN, AMBER],
                marker_line_width=0,
                text=[f"{v:.1f}%" for v in sub_cl["rate"]],
                textposition="outside",
            )
        )

        fig_clb.update_layout(
            **{
                **PLOT_LAYOUT,
                "title": "Subscription Rate per Cluster",
                "xaxis": dict(showgrid=False, title="Cluster"),
                "yaxis": dict(
                    showgrid=True,
                    gridcolor="#f0f2f5",
                    title="Rate (%)",
                ),
            }
        )
    else:
        fig_clb = go.Figure()

    seg_cards = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.Div(
                            "🟢 Cluster 0",
                            style={
                                "fontSize": "11px",
                                "fontWeight": "600",
                                "color": GREEN,
                                "textTransform": "uppercase",
                                "letterSpacing": ".06em",
                            },
                        ),
                        html.Div(
                            "High-Value",
                            style={
                                "fontSize": "15px",
                                "fontWeight": "600",
                                "marginTop": "4px",
                            },
                        ),
                        html.Div(
                            "Moderate age, prior engagement. Most likely to subscribe.",
                            style={
                                "fontSize": "12px",
                                "color": "#4a5568",
                                "lineHeight": "1.5",
                                "marginTop": "6px",
                            },
                        ),
                    ],
                    className="seg-card",
                    style={"background": "#EAF3DE"},
                ),
                md=4,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div(
                            "🔵 Cluster 1",
                            style={
                                "fontSize": "11px",
                                "fontWeight": "600",
                                "color": BLUE,
                                "textTransform": "uppercase",
                                "letterSpacing": ".06em",
                            },
                        ),
                        html.Div(
                            "Nurture",
                            style={
                                "fontSize": "15px",
                                "fontWeight": "600",
                                "marginTop": "4px",
                            },
                        ),
                        html.Div(
                            "Minimal interaction history. Needs awareness-building.",
                            style={
                                "fontSize": "12px",
                                "color": "#4a5568",
                                "lineHeight": "1.5",
                                "marginTop": "6px",
                            },
                        ),
                    ],
                    className="seg-card",
                    style={"background": "#E6F1FB"},
                ),
                md=4,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div(
                            "🟡 Cluster 2",
                            style={
                                "fontSize": "11px",
                                "fontWeight": "600",
                                "color": AMBER,
                                "textTransform": "uppercase",
                                "letterSpacing": ".06em",
                            },
                        ),
                        html.Div(
                            "Over-Targeted",
                            style={
                                "fontSize": "15px",
                                "fontWeight": "600",
                                "marginTop": "4px",
                            },
                        ),
                        html.Div(
                            "High contacts, low response. Reduce frequency and redesign messaging.",
                            style={
                                "fontSize": "12px",
                                "color": "#4a5568",
                                "lineHeight": "1.5",
                                "marginTop": "6px",
                            },
                        ),
                    ],
                    className="seg-card",
                    style={"background": "#FAEEDA"},
                ),
                md=4,
            ),
        ],
        className="mb-3",
    )

    return html.Div(
        [
            html.Div("Customer Segments", className="page-title"),

            dbc.Row(
                [
                    dbc.Col(
                        card([dcc.Graph(figure=fig_elbow, config={"displayModeBar": False})]),
                        md=6,
                    ),
                    dbc.Col(
                        card([dcc.Graph(figure=fig_sc, config={"displayModeBar": False})]),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),

            seg_cards,

            card(
                [
                    section_header("Cluster Profiles"),
                    dash_table.DataTable(
                        data=profile.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in profile.columns],
                        style_table={
                            "borderRadius": "8px",
                            "overflow": "hidden",
                        },
                        style_header={
                            "backgroundColor": LIGHT,
                            "fontWeight": "600",
                            "fontSize": "12px",
                            "color": DARK,
                            "border": f"1px solid {BORDER}",
                        },
                        style_cell={
                            "fontFamily": "DM Sans",
                            "fontSize": "13px",
                            "padding": "10px 14px",
                            "border": f"1px solid {BORDER}",
                            "textAlign": "center",
                        },
                    ),
                ],
                style={"marginBottom": "16px"},
            ),

            dbc.Row(
                [
                    dbc.Col(
                        card([dcc.Graph(figure=fig_clb, config={"displayModeBar": False})]),
                        md=12,
                    )
                ]
            ),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run app
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
