"""
Telephony Transfer Pathway Dashboard
Focused on master_contact_id transfer journeys and performance impact.
"""

from __future__ import annotations

import os
import io
import base64
from typing import Iterable

import dash
from dash import dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False
    plt = None
    sns = None


DATA_PATH = os.path.expanduser(
    os.environ.get("TELEPHONY_SAMPLE_PATH", "~/Downloads/telephony_transfer_sample.csv")
)
MIN_REAL_ROWS_FOR_ANALYSIS = 120
DEFAULT_SYNTHETIC_JOURNEYS = 900


def load_calls(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "master_contact_id": "string",
            "contact_id": "string",
            "POLICYNUMBER": "string",
            "ACCOUNT_NUMBER": "string",
        },
    )
    df.columns = [c.strip() for c in df.columns]
    df["CALL_START_WITH_COLLEAGUE"] = pd.to_datetime(
        df["CALL_START_WITH_COLLEAGUE"], dayfirst=True, errors="coerce"
    )
    df["DOB"] = pd.to_datetime(df["DOB"], dayfirst=True, errors="coerce")

    for c in [
        "CONTACT_ID_DURATION",
        "HANDLETIME",
        "TALKTIME",
        "WRAPTIME",
        "SKILL_COUNT",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["INBOUND_OUTBOUND"] = df["INBOUND_OUTBOUND"].astype("string").str.upper().str.strip()
    df["skill_name"] = df["skill_name"].astype("string").str.strip()
    df["LINE_OF_BUSINESS"] = df["LINE_OF_BUSINESS"].astype("string").str.strip()

    df = df.sort_values(["master_contact_id", "CALL_START_WITH_COLLEAGUE", "contact_id"])
    return df


def generate_synthetic_calls(
    n_journeys: int = DEFAULT_SYNTHETIC_JOURNEYS, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skills = [
        "HD RTL CS",
        "HD RTL CS YOUDRIVE",
        "HD RTL CS RENEWALS",
        "HD RTL CS COMPLAINTS",
        "HD RTL CS NEW BUSINESS",
        "HD RTL CS RETENTION",
        "HD RTL CS CLAIMS",
        "HD RTL CS PAYMENTS",
        "HD RTL CS BILLING",
    ]
    lobs = ["Car", "Home", "Bike", "Van", "MultiCar"]
    directions = ["INBOUND", "OUTBOUND"]
    base_start = pd.Timestamp("2025-01-01")

    rows = []
    contact_counter = 500000
    for i in range(n_journeys):
        master_id = f"SYN-{100000 + i}"
        lob = rng.choice(lobs, p=[0.45, 0.2, 0.12, 0.1, 0.13])
        direction = rng.choice(directions, p=[0.78, 0.22])

        transfer_count = int(rng.choice([0, 1, 2, 3, 4], p=[0.5, 0.24, 0.15, 0.08, 0.03]))
        skill_count = transfer_count + 1

        start = base_start + pd.Timedelta(days=int(rng.integers(0, 365)))
        chosen = rng.choice(skills, size=max(2, skill_count), replace=False).tolist()
        route = [chosen[0]]
        for step in range(1, skill_count):
            # Inject loop behavior for a subset of multi-transfer journeys
            if step >= 2 and rng.random() < 0.18:
                route.append(route[step - 2])
            else:
                route.append(chosen[min(step, len(chosen) - 1)])

        complexity_multiplier = 1 + (transfer_count * 0.42)
        if direction == "OUTBOUND":
            complexity_multiplier *= 0.9

        for idx, skill in enumerate(route, start=1):
            contact_counter += 1
            step_start = start + pd.Timedelta(minutes=int(idx * rng.integers(3, 18)))
            talk = max(1, int(rng.normal(8, 2) * complexity_multiplier))
            wrap = max(1, int(rng.normal(4, 1.3) * complexity_multiplier))
            handle = talk + wrap + max(1, int(rng.normal(3, 1.1)))
            dur = handle + max(0, int(rng.normal(1.2, 0.8)))
            dob_year = int(rng.integers(1958, 2005))
            dob = pd.Timestamp(f"{dob_year}-{int(rng.integers(1,13)):02d}-{int(rng.integers(1,28)):02d}")

            rows.append(
                {
                    "master_contact_id": master_id,
                    "contact_id": str(contact_counter),
                    "skill_name": skill,
                    "INBOUND_OUTBOUND": direction,
                    "CALL_START_WITH_COLLEAGUE": step_start,
                    "CONTACT_ID_DURATION": dur,
                    "HANDLETIME": handle,
                    "TALKTIME": talk,
                    "WRAPTIME": wrap,
                    "SKILL_COUNT": skill_count,
                    "POLICYNUMBER": str(7000000 + int(rng.integers(0, 999999))),
                    "ACCOUNT_NUMBER": str(4000000 + int(rng.integers(0, 999999))),
                    "DOB": dob,
                    "LINE_OF_BUSINESS": lob,
                }
            )

    return pd.DataFrame(rows)


def combine_with_synthetic_if_needed(real_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if len(real_df) >= MIN_REAL_ROWS_FOR_ANALYSIS:
        return real_df.copy(), "real_only"
    needed = max(DEFAULT_SYNTHETIC_JOURNEYS, MIN_REAL_ROWS_FOR_ANALYSIS - len(real_df) + 300)
    synth = generate_synthetic_calls(n_journeys=needed, seed=42)
    combined = pd.concat([real_df, synth], ignore_index=True)
    combined = combined.sort_values(["master_contact_id", "CALL_START_WITH_COLLEAGUE", "contact_id"])
    return combined, f"real_plus_synthetic_{needed}"


def _mode_or_mixed(values: Iterable[str]) -> str:
    s = pd.Series(values).dropna()
    if s.empty:
        return "Unknown"
    uniq = s.unique()
    return uniq[0] if len(uniq) == 1 else "MIXED"


def build_journeys(calls: pd.DataFrame) -> pd.DataFrame:
    def make_path(g: pd.DataFrame) -> str:
        ordered = g.sort_values("CALL_START_WITH_COLLEAGUE")
        return " -> ".join(ordered["skill_name"].fillna("Unknown").tolist())

    def has_loop(g: pd.DataFrame) -> int:
        return int(g["skill_name"].duplicated().any())

    j = (
        calls.groupby("master_contact_id", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "started_at": g["CALL_START_WITH_COLLEAGUE"].min(),
                    "line_of_business": _mode_or_mixed(g["LINE_OF_BUSINESS"]),
                    "direction": _mode_or_mixed(g["INBOUND_OUTBOUND"]),
                    "contact_rows": int(g["contact_id"].nunique()),
                    "unique_skills": int(g["skill_name"].nunique()),
                    "skill_count_max": float(g["SKILL_COUNT"].max()),
                    "handle_total": float(g["HANDLETIME"].sum()),
                    "talk_total": float(g["TALKTIME"].sum()),
                    "wrap_total": float(g["WRAPTIME"].sum()),
                    "duration_total": float(g["CONTACT_ID_DURATION"].sum()),
                    "loop_flag": has_loop(g),
                    "pathway": make_path(g),
                }
            )
        )
        .reset_index()
    )

    j["transfer_count_skill"] = (j["skill_count_max"].fillna(1) - 1).clip(lower=0)
    j["transfer_count_rows"] = (j["contact_rows"] - 1).clip(lower=0)
    j["transfer_count"] = j["transfer_count_skill"].round().astype(int)
    j["transfer_definition_mismatch"] = (
        j["transfer_count"] != j["transfer_count_rows"].astype(int)
    ).astype(int)
    j["transferred"] = (j["transfer_count"] > 0).astype(int)
    j["multi_transfer"] = (j["transfer_count"] >= 2).astype(int)
    j["aht_per_contact"] = np.where(
        j["contact_rows"] > 0, j["handle_total"] / j["contact_rows"], np.nan
    )
    j["date"] = j["started_at"].dt.date
    return j


def build_transitions(calls: pd.DataFrame) -> pd.DataFrame:
    links = []
    for _, g in calls.groupby("master_contact_id", dropna=False):
        skills = g.sort_values("CALL_START_WITH_COLLEAGUE")["skill_name"].fillna("Unknown").tolist()
        if len(skills) < 2:
            continue
        for i in range(len(skills) - 1):
            links.append({"source": skills[i], "target": skills[i + 1]})
    if not links:
        return pd.DataFrame(columns=["source", "target", "count"])
    out = pd.DataFrame(links).value_counts().reset_index(name="count")
    return out


real_calls_df = load_calls(DATA_PATH)
calls_df, dataset_mode = combine_with_synthetic_if_needed(real_calls_df)
journey_df = build_journeys(calls_df)
min_date = journey_df["started_at"].min().date() if not journey_df.empty else pd.Timestamp.today().date()
max_date = journey_df["started_at"].max().date() if not journey_df.empty else pd.Timestamp.today().date()
lob_values = sorted([x for x in journey_df["line_of_business"].dropna().unique().tolist() if x != ""])
queue_values = sorted([x for x in calls_df["skill_name"].dropna().unique().tolist() if x != ""])


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Telephonyt Transfer Analyttics"

POWERBI_COLORS = {
    "primary": "#00BCF2",
    "secondary": "#742774",
    "success": "#00A86B",
    "warning": "#FFB900",
    "danger": "#E81123",
    "dark": "#252423",
    "light": "#F3F2F1",
}
GRAPH_CONFIG = {"responsive": True, "displayModeBar": False}

DT_STYLE_HEADER = {
    "backgroundColor": "#0078D4",
    "color": "white",
    "fontWeight": "700",
    "fontSize": "0.75rem",
    "textTransform": "uppercase",
    "letterSpacing": "0.4px",
    "border": "none",
    "padding": "10px 12px",
    "fontFamily": "Segoe UI, sans-serif",
}
DT_STYLE_DATA = {
    "backgroundColor": "white",
    "color": "#201F1E",
    "fontSize": "0.83rem",
    "fontFamily": "Segoe UI, sans-serif",
    "border": "1px solid #EDEBE9",
    "padding": "8px 12px",
}
DT_STYLE_CONDITIONAL = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#F8F8F8"},
    {
        "if": {"state": "selected"},
        "backgroundColor": "rgba(0,188,242,0.08)",
        "border": "1px solid #00BCF2",
    },
]


def filter_journeys(start_date, end_date, direction, lobs, queues):
    df = journey_df.copy()
    if start_date and end_date:
        s = pd.to_datetime(start_date).date()
        e = pd.to_datetime(end_date).date()
        df = df[(df["date"] >= s) & (df["date"] <= e)]
    if direction and direction != "BOTH":
        df = df[df["direction"] == direction]
    if lobs:
        df = df[df["line_of_business"].isin(lobs)]
    if queues:
        q_ids = calls_df[calls_df["skill_name"].isin(queues)]["master_contact_id"].dropna().unique().tolist()
        df = df[df["master_contact_id"].isin(q_ids)]
    return df


def filter_calls_by_journeys(filtered_journeys: pd.DataFrame) -> pd.DataFrame:
    ids = filtered_journeys["master_contact_id"].dropna().unique().tolist()
    return calls_df[calls_df["master_contact_id"].isin(ids)].copy()


def graph_component(fig: go.Figure, height: int = 520) -> dcc.Graph:
    fig.update_layout(
        autosize=True,
        height=height,
        margin=dict(l=40, r=20, t=60, b=50),
    )
    return dcc.Graph(
        figure=fig,
        config=GRAPH_CONFIG,
        style={"width": "100%", "height": f"{height}px"},
    )


def seaborn_img_component(fig, alt_text: str = "", height: int = 520):
    if not HAS_SEABORN:
        return seaborn_fallback("Seaborn/Matplotlib not available in this environment.")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        alt=alt_text,
        style={
            "width": "100%",
            "height": f"{height}px",
            "objectFit": "contain",
            "backgroundColor": "white",
            "boxShadow": "var(--shadow-sm)",
            "borderRadius": "6px",
            "padding": "8px",
        },
    )


def seaborn_fallback(message: str):
    return html.Div()


app.layout = dbc.Container(
    [
        html.Div(
            "HAMZAH JAVAID",
            style={
                "position": "fixed",
                "top": "48%",
                "left": "50%",
                "transform": "translate(-50%, -50%) rotate(-24deg)",
                "fontSize": "5.2rem",
                "fontWeight": "800",
                "letterSpacing": "6px",
                "color": "rgba(32,31,30,0.06)",
                "zIndex": "0",
                "pointerEvents": "none",
                "userSelect": "none",
                "whiteSpace": "nowrap",
            },
        ),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="/assets/hastings_logo.svg",
                        style={"height": "44px", "objectFit": "contain"},
                        title="Hastings Direct",
                    ),
                ], xs="auto", className="d-flex align-items-center"),
                dbc.Col([
                    html.Div([
                        html.H1(
                            "Telephonyt Transfer Analyttics",
                            style={
                                "color": POWERBI_COLORS["primary"],
                                "fontWeight": "700",
                                "fontSize": "1.8rem",
                                "marginBottom": "0.15rem",
                                "lineHeight": "1.2",
                            },
                        ),
                        html.P(
                            "Telephony transfer pathways, effort impact, and routing patterns",
                            style={"color": "#605E5C", "fontSize": "0.88rem", "marginBottom": "0"},
                        ),
                        html.P(
                            "Built by HAMZAH JAVAID",
                            style={"color": "#8A8886", "fontSize": "0.75rem", "marginBottom": "0"},
                        ),
                    ])
                ], className="d-flex align-items-center"),
                dbc.Col([
                    html.Div([
                        html.Div(
                            "INTERNAL USE ONLY",
                            style={
                                "fontSize": "0.65rem",
                                "fontWeight": "700",
                                "letterSpacing": "1.2px",
                                "color": "#A0A0A0",
                                "textTransform": "uppercase",
                                "textAlign": "right",
                            },
                        ),
                        html.Div(
                            "Customer Operations",
                            style={"fontSize": "0.75rem", "color": "#888", "textAlign": "right"},
                        ),
                        html.Div(
                            "HAMZAH JAVAID",
                            style={
                                "fontSize": "0.72rem",
                                "fontWeight": "700",
                                "letterSpacing": "0.7px",
                                "color": "#9A9896",
                                "textAlign": "right",
                            },
                        ),
                    ])
                ], xs="auto", className="d-flex align-items-center ms-auto"),
            ], align="center", className="g-3"),
            html.Hr(className="divider", style={"marginTop": "0.9rem", "marginBottom": "0"}),
        ], className="mb-3", style={"paddingTop": "0.5rem"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "REPORT FILTERS",
                            style={
                                "fontSize": "0.7rem",
                                "fontWeight": "700",
                                "color": "#777",
                                "letterSpacing": "1.3px",
                            },
                        )
                    ],
                    style={
                        "paddingBottom": "10px",
                        "marginBottom": "12px",
                        "borderBottom": "1px solid #E0E0E0",
                    },
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "DATE RANGE",
                                                style={
                                                    "fontSize": "0.7rem",
                                                    "fontWeight": "700",
                                                    "color": "#444",
                                                    "letterSpacing": "0.5px",
                                                },
                                            )
                                        ],
                                        className="slicer-header",
                                    ),
                                    html.Div(
                                        [
                                            dcc.DatePickerRange(
                                                id="date-filter",
                                                min_date_allowed=min_date,
                                                max_date_allowed=max_date,
                                                start_date=min_date,
                                                end_date=max_date,
                                                display_format="DD/MM/YYYY",
                                                style={"fontSize": "0.82rem"},
                                            )
                                        ],
                                        className="slicer-body",
                                    ),
                                ],
                                className="slicer-card",
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "DIRECTION",
                                                style={
                                                    "fontSize": "0.7rem",
                                                    "fontWeight": "700",
                                                    "color": "#444",
                                                    "letterSpacing": "0.5px",
                                                },
                                            )
                                        ],
                                        className="slicer-header",
                                    ),
                                    html.Div(
                                        [
                                            dbc.RadioItems(
                                                id="direction-filter",
                                                options=[
                                                    {"label": "Both", "value": "BOTH"},
                                                    {"label": "Inbound", "value": "INBOUND"},
                                                    {"label": "Outbound", "value": "OUTBOUND"},
                                                ],
                                                value="BOTH",
                                                inline=True,
                                            )
                                        ],
                                        className="slicer-body",
                                    ),
                                ],
                                className="slicer-card",
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "LINE OF BUSINESS",
                                                style={
                                                    "fontSize": "0.7rem",
                                                    "fontWeight": "700",
                                                    "color": "#444",
                                                    "letterSpacing": "0.5px",
                                                },
                                            )
                                        ],
                                        className="slicer-header",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="lob-filter",
                                                options=[{"label": x, "value": x} for x in lob_values],
                                                value=lob_values,
                                                multi=True,
                                                placeholder="Line of business",
                                                style={"fontSize": "0.82rem"},
                                            )
                                        ],
                                        className="slicer-body",
                                    ),
                                ],
                                className="slicer-card",
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "QUEUE / SKILL",
                                                style={
                                                    "fontSize": "0.7rem",
                                                    "fontWeight": "700",
                                                    "color": "#444",
                                                    "letterSpacing": "0.5px",
                                                },
                                            )
                                        ],
                                        className="slicer-header",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="queue-filter",
                                                options=[{"label": q, "value": q} for q in queue_values],
                                                value=queue_values,
                                                multi=True,
                                                placeholder="Select skills",
                                                style={"fontSize": "0.82rem"},
                                            )
                                        ],
                                        className="slicer-body",
                                    ),
                                ],
                                className="slicer-card",
                            ),
                            md=3,
                        ),
                    ],
                    className="g-3",
                ),
            ],
            className="filter-panel mb-3",
        ),
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview & Definitions", value="overview"),
                dcc.Tab(label="Process & Routing", value="process"),
                dcc.Tab(label="Journey Pathways", value="pathways"),
                dcc.Tab(label="Cost & Effort Impact", value="impact"),
                dcc.Tab(label="Hours & Transfer Effect", value="hours"),
                dcc.Tab(label="Uplift & Leakage", value="uplift"),
                dcc.Tab(label="Data Explorer", value="explorer"),
            ],
        ),
        html.Div(id="tab-content", className="mt-3"),
    ],
    fluid=True,
)


@callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("direction-filter", "value"),
    Input("lob-filter", "value"),
    Input("queue-filter", "value"),
)
def render_tab(tab, start_date, end_date, direction, lobs, queues):
    filtered_j = filter_journeys(start_date, end_date, direction, lobs or [], queues or [])
    filtered_c = filter_calls_by_journeys(filtered_j)

    if filtered_j.empty:
        return dbc.Alert("No journeys for selected filters.", color="warning")

    transfer_rate = filtered_j["transferred"].mean() * 100
    multi_transfer_rate = (filtered_j["transfer_count"] >= 2).mean() * 100
    loop_rate = filtered_j["loop_flag"].mean() * 100
    avg_transfers = filtered_j.loc[filtered_j["transferred"] == 1, "transfer_count"].mean()
    avg_transfers = 0 if pd.isna(avg_transfers) else avg_transfers
    mismatch_rate = filtered_j["transfer_definition_mismatch"].mean() * 100

    if tab == "overview":
        contents = [
            ("Process & Routing", "Entry skill risk, handoff patterns, and transition intensity."),
            ("Journey Pathways", "Forward/backward Sankey views and top complete pathway table."),
            ("Cost & Effort Impact", "How transfers increase handle, talk, wrap, and AHT cost curves."),
            ("Hours & Transfer Effect", "Transfer and effort patterns by hour/day and transfer band."),
            ("Uplift & Leakage", "Excess effort vs baseline, leakage pathways, and outlier diagnostics."),
            ("Data Explorer", "Row-level data inspection with filter/sort for investigations."),
        ]
        content_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(name, style={"fontWeight": "700"}),
                                html.P(desc, style={"marginBottom": "0", "fontSize": "0.9rem"}),
                            ]
                        )
                    ),
                    md=6,
                    className="mb-3",
                )
                for name, desc in contents
            ]
        )

        defs = pd.DataFrame(
            [
                {"Metric": "Transfer Rate", "Definition": "% of journeys with transfer_count > 0."},
                {"Metric": "Multi-Transfer Rate", "Definition": "% of journeys with transfer_count >= 2."},
                {"Metric": "Loop Rate", "Definition": "% of journeys where a skill repeats (e.g., A -> B -> A)."},
                {"Metric": "Handle Uplift", "Definition": "Median handle for a transfer band relative to zero-transfer median."},
                {"Metric": "Leakage Path", "Definition": "Pathway with high excess handle vs expected baseline effort."},
                {"Metric": "Excess Handle", "Definition": "Observed handle minus expected baseline handle for the same volume."},
                {"Metric": "Wrap/Talk Ratio", "Definition": "WRAPTIME divided by TALKTIME; higher values may indicate process friction."},
                {"Metric": "Mismatch Rate", "Definition": "% where SKILL_COUNT-1 does not match observed row-based transfers."},
            ]
        )
        defs_table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in defs.columns],
            data=defs.to_dict("records"),
            style_header=DT_STYLE_HEADER,
            style_data=DT_STYLE_DATA,
            style_data_conditional=DT_STYLE_CONDITIONAL,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto"},
            page_action="none",
        )

        return html.Div(
            [
                html.H5("Dashboard Contents", style={"fontWeight": "700"}),
                html.P("Use this page as a guide to what each tab covers.", className="text-muted"),
                content_cards,
                html.Hr(className="divider"),
                html.H5("Definitions", style={"fontWeight": "700"}),
                html.P("Core metrics and formulas used across the dashboard.", className="text-muted"),
                defs_table,
            ]
        )

    if tab == "process":
        entry_skill = (
            filtered_c.sort_values(["master_contact_id", "CALL_START_WITH_COLLEAGUE"])
            .groupby("master_contact_id")
            .first()
            .reset_index()
        )
        entry_perf = (
            entry_skill.merge(filtered_j[["master_contact_id", "transferred"]], on="master_contact_id", how="left")
            .groupby("skill_name")
            .agg(journeys=("master_contact_id", "count"), transfer_rate=("transferred", "mean"))
            .reset_index()
            .sort_values("journeys", ascending=False)
            .head(12)
        )
        entry_perf["transfer_rate"] = entry_perf["transfer_rate"] * 100
        entry_perf = entry_perf.sort_values("transfer_rate", ascending=False).head(15)

        trans = build_transitions(filtered_c)
        pair_tbl = pd.DataFrame(columns=["source", "target", "count", "share_pct"])
        if not trans.empty:
            trans = trans.sort_values("count", ascending=False)
            trans["share_pct"] = (trans["count"] / trans["count"].sum() * 100).round(2)
            pair_tbl = trans.head(20)

        if not HAS_SEABORN:
            entry_fig = px.bar(entry_perf, y="skill_name", x="transfer_rate", orientation="h", title="Entry Skill Transfer Rate")
            return html.Div([seaborn_fallback("Seaborn not available; using Plotly fallback."), graph_component(entry_fig, 540)])

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=entry_perf, y="skill_name", x="transfer_rate", color="#742774", ax=ax1)
        ax1.set_title("Entry Skill Transfer Risk Ranking")
        ax1.set_xlabel("Transfer Rate %")
        ax1.set_ylabel("Entry Skill")

        mat = pd.DataFrame(index=["No Data"], columns=["No Data"], data=0)
        if not trans.empty:
            top_nodes = list(pd.concat([trans["source"], trans["target"]]).value_counts().head(10).index)
            mat = (
                trans[trans["source"].isin(top_nodes) & trans["target"].isin(top_nodes)]
                .pivot(index="source", columns="target", values="count")
                .fillna(0)
            )
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.heatmap(mat, cmap="mako", annot=False, ax=ax2, cbar_kws={"label": "Transition Count"})
        ax2.set_title("Skill-to-Skill Transition Intensity (Top Nodes)")
        ax2.set_xlabel("To Skill")
        ax2.set_ylabel("From Skill")

        pair_table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in pair_tbl.columns],
            data=pair_tbl.to_dict("records"),
            page_size=10,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header=DT_STYLE_HEADER,
            style_data=DT_STYLE_DATA,
            style_data_conditional=DT_STYLE_CONDITIONAL,
        )
        return html.Div(
            [
                seaborn_img_component(fig1, "entry skill risk ranking", 560),
                seaborn_img_component(fig2, "transition heatmap", 580),
                html.H6("High-Volume Handoff Pairs"),
                pair_table,
            ]
        )

    if tab == "pathways":
        all_skills = sorted(filtered_c["skill_name"].dropna().unique().tolist())
        return html.Div(
            [
                html.H5(
                    "Customer Journey Pathways",
                    style={"fontWeight": "700", "color": "#201F1E", "marginBottom": "0.3rem"},
                ),
                html.P(
                    "Visualise how contacts flow through skills: forward paths (where they go) and backward paths (how they arrived).",
                    className="text-muted mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "SELECT SKILL TO ANALYSE",
                                                style={
                                                    "fontSize": "0.7rem",
                                                    "fontWeight": "700",
                                                    "color": "#444",
                                                    "letterSpacing": "0.5px",
                                                },
                                            )
                                        ],
                                        className="slicer-header",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="journey-queue-selector",
                                                options=[{"label": q, "value": q} for q in all_skills],
                                                value=all_skills[0] if all_skills else None,
                                                placeholder="Choose a skill...",
                                                clearable=False,
                                                style={"fontSize": "0.9rem"},
                                            )
                                        ],
                                        className="slicer-body",
                                    ),
                                ],
                                className="slicer-card",
                            ),
                            md=5,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "JOURNEY DEPTH (LEVELS)",
                                                style={
                                                    "fontSize": "0.7rem",
                                                    "fontWeight": "700",
                                                    "color": "#444",
                                                    "letterSpacing": "0.5px",
                                                },
                                            )
                                        ],
                                        className="slicer-header",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Slider(
                                                id="journey-depth-slider",
                                                min=2,
                                                max=5,
                                                value=3,
                                                marks={i: str(i) for i in range(2, 6)},
                                                tooltip={"placement": "bottom", "always_visible": True},
                                            )
                                        ],
                                        className="slicer-body",
                                        style={"paddingTop": "12px"},
                                    ),
                                ],
                                className="slicer-card",
                            ),
                            md=5,
                        ),
                    ],
                    className="mb-4",
                ),
                html.Div(id="journey-analysis"),
            ]
        )

    if tab == "impact":
        summary = (
            filtered_j.groupby("transfer_count")
            .agg(
                journeys=("master_contact_id", "count"),
                avg_handle=("handle_total", "mean"),
                avg_talk=("talk_total", "mean"),
                avg_wrap=("wrap_total", "mean"),
                avg_aht=("aht_per_contact", "mean"),
            )
            .reset_index()
            .round(2)
        )
        tbl = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in summary.columns],
            data=summary.to_dict("records"),
            page_size=10,
            style_table={"overflowX": "auto"},
        )
        no_xfer = filtered_j[filtered_j["transfer_count"] == 0]["handle_total"].median()
        heavy_xfer = filtered_j[filtered_j["transfer_count"] >= 2]["handle_total"].median()
        uplift_pct = ((heavy_xfer / no_xfer) - 1) * 100 if no_xfer and not np.isnan(no_xfer) else np.nan
        uplift_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Multi-Transfer Handle Uplift"),
                    html.H4("n/a" if np.isnan(uplift_pct) else f"{uplift_pct:.1f}%"),
                    html.Small("Median handle time: transfer_count >=2 vs 0"),
                ]
            ),
            className="mb-3",
        )
        if HAS_SEABORN:
            summary_long = summary.melt(id_vars=["transfer_count"], value_vars=["avg_handle", "avg_talk", "avg_wrap", "avg_aht"], var_name="metric", value_name="value")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=summary_long, x="transfer_count", y="value", hue="metric", marker="o", ax=ax1)
            ax1.set_title("Cost Curve by Transfer Count")
            ax1.set_xlabel("Transfer Count")
            ax1.set_ylabel("Minutes")

            summary["marginal_handle_delta"] = summary["avg_handle"].diff().fillna(0)
            fig2, ax2 = plt.subplots(figsize=(12, 4.6))
            sns.barplot(data=summary, x="transfer_count", y="marginal_handle_delta", color="#E81123", ax=ax2)
            ax2.axhline(0, color="#666", linewidth=1)
            ax2.set_title("Marginal Handle Cost per Additional Transfer Step")
            ax2.set_xlabel("Transfer Count")
            ax2.set_ylabel("Delta Minutes vs Previous Band")
            main_vis = [seaborn_img_component(fig1, "cost curve by transfer count", 560), seaborn_img_component(fig2, "marginal transfer cost", 470)]
        else:
            melted = filtered_j.melt(
                id_vars=["transfer_count"],
                value_vars=["handle_total", "talk_total", "wrap_total", "aht_per_contact"],
                var_name="metric",
                value_name="minutes",
            )
            fig = px.box(melted, x="transfer_count", y="minutes", color="metric", title="Effort Distribution by Transfer Count")
            main_vis = [seaborn_fallback("Seaborn not available; using Plotly fallback."), graph_component(fig, 560)]
        return html.Div([uplift_card, *main_vis, html.H5("Impact Summary"), tbl])

    if tab == "hours":
        temp = filtered_c.copy()
        temp["hour"] = temp["CALL_START_WITH_COLLEAGUE"].dt.hour
        temp["weekday"] = temp["CALL_START_WITH_COLLEAGUE"].dt.day_name()
        temp = temp.merge(
            filtered_j[["master_contact_id", "transfer_count", "transferred"]],
            on="master_contact_id",
            how="left",
        )
        temp["transfer_band"] = pd.cut(
            temp["transfer_count"],
            bins=[-0.1, 0, 1, 2, 999],
            labels=["0", "1", "2", "3+"],
        )
        hm = (
            temp.groupby(["hour", "transfer_band"], observed=True)["HANDLETIME"]
            .median()
            .reset_index(name="median_handle")
        )
        if HAS_SEABORN:
            heat1 = temp.groupby(["weekday", "hour"])["transferred"].mean().mul(100).reset_index(name="transfer_rate")
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heat1["weekday"] = pd.Categorical(heat1["weekday"], categories=weekday_order, ordered=True)
            pivot1 = heat1.pivot(index="weekday", columns="hour", values="transfer_rate").sort_index()
            fig1, ax1 = plt.subplots(figsize=(12, 4.8))
            sns.heatmap(pivot1, cmap="rocket_r", ax=ax1, cbar_kws={"label": "Transfer Rate %"})
            ax1.set_title("Transfer Rate by Weekday and Hour")
            ax1.set_xlabel("Hour")
            ax1.set_ylabel("Weekday")

            pivot2 = hm.pivot(index="transfer_band", columns="hour", values="median_handle").fillna(0)
            fig2, ax2 = plt.subplots(figsize=(12, 4.8))
            sns.heatmap(pivot2, cmap="viridis", ax=ax2, cbar_kws={"label": "Median Handle (min)"})
            ax2.set_title("Median Handle Time by Hour and Transfer Band")
            ax2.set_xlabel("Hour")
            ax2.set_ylabel("Transfer Band")

            return html.Div([seaborn_img_component(fig1, "transfer rate by weekday and hour", 500), seaborn_img_component(fig2, "median handle by hour and transfer band", 500)])
        fig_hm = px.density_heatmap(hm, x="hour", y="transfer_band", z="median_handle", title="Median Handle Time by Hour and Transfer Band")
        return html.Div([seaborn_fallback("Seaborn not available; using Plotly fallback."), graph_component(fig_hm, 540)])

    if tab == "uplift":
        base = filtered_j[filtered_j["transferred"] == 1].copy()
        if base.empty:
            return dbc.Alert("No transferred journeys in this filter.", color="info")

        baseline = filtered_j.loc[filtered_j["transfer_count"] == 0, "handle_total"].median()
        p95 = base["handle_total"].quantile(0.95)
        mystery = base[(base["handle_total"] >= p95) | (base["loop_flag"] == 1)].copy()
        mystery["wrap_to_talk_ratio"] = np.where(mystery["talk_total"] > 0, mystery["wrap_total"] / mystery["talk_total"], np.nan)
        mystery["mystery_tag"] = np.select(
            [
                (mystery["loop_flag"] == 1) & (mystery["handle_total"] >= p95),
                mystery["loop_flag"] == 1,
                mystery["handle_total"] >= p95,
                mystery["wrap_to_talk_ratio"] >= 0.7,
            ],
            ["Loop + High Handle", "Loop", "High Handle", "High Wrap Ratio"],
            default="Check",
        )
        mystery = mystery.sort_values(["handle_total", "transfer_count"], ascending=False)

        uplift_tbl = (
            filtered_j.groupby("transfer_count")
            .agg(
                journeys=("master_contact_id", "count"),
                median_handle=("handle_total", "median"),
                mean_handle=("handle_total", "mean"),
            )
            .reset_index()
            .sort_values("transfer_count")
        )
        uplift_tbl["uplift_vs_baseline_pct"] = np.where(
            baseline > 0,
            ((uplift_tbl["median_handle"] / baseline) - 1) * 100,
            np.nan,
        )

        path_leak = (
            filtered_j.groupby("pathway")
            .agg(
                journeys=("master_contact_id", "count"),
                total_handle=("handle_total", "sum"),
                avg_handle=("handle_total", "mean"),
                avg_transfers=("transfer_count", "mean"),
            )
            .reset_index()
        )
        path_leak["expected_handle_baseline"] = path_leak["journeys"] * (baseline if not pd.isna(baseline) else 0)
        path_leak["excess_handle"] = path_leak["total_handle"] - path_leak["expected_handle_baseline"]
        path_leak["leakage_share_pct"] = (path_leak["total_handle"] / max(path_leak["total_handle"].sum(), 1) * 100).round(2)
        path_leak = path_leak.sort_values("excess_handle", ascending=False).head(15)

        uplift_fig = px.bar(
            uplift_tbl,
            x="transfer_count",
            y="uplift_vs_baseline_pct",
            text=uplift_tbl["uplift_vs_baseline_pct"].round(1),
            title="Uplift Curve: Median Handle % vs Zero-Transfer Baseline",
        )
        uplift_fig.update_traces(marker_color="#E81123")

        leak_fig = px.bar(
            path_leak,
            y="pathway",
            x="excess_handle",
            orientation="h",
            hover_data=["journeys", "avg_handle", "avg_transfers", "leakage_share_pct"],
            title="Leakage Paths: Excess Handle Minutes vs Baseline",
        )
        leak_fig.update_layout(yaxis={"categoryorder": "total ascending"})

        scatter = px.scatter(
            mystery,
            x="transfer_count",
            y="handle_total",
            color="mystery_tag",
            hover_data=["master_contact_id", "pathway", "line_of_business", "wrap_to_talk_ratio"],
            title="Outlier Diagnostics: High-Cost Transferred Journeys",
        )
        scatter.add_hline(y=p95, line_dash="dash", annotation_text="95th percentile handle time")

        mystery_view = mystery[
            [
                "master_contact_id",
                "date",
                "direction",
                "line_of_business",
                "transfer_count",
                "loop_flag",
                "handle_total",
                "talk_total",
                "wrap_total",
                "wrap_to_talk_ratio",
                "mystery_tag",
                "aht_per_contact",
                "pathway",
            ]
        ].head(100)
        tbl = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in mystery_view.columns],
            data=mystery_view.to_dict("records"),
            page_size=10,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "fontSize": "12px"},
        )
        explain = html.Div(
            [
                html.P(
                    [
                        html.Strong("Uplift definition: "),
                        "For each transfer band, uplift is median handle time relative to the zero-transfer baseline.",
                    ],
                    style={"marginBottom": "0.2rem"},
                ),
                html.P(
                    [
                        html.Strong("Leakage definition: "),
                        "For each pathway, excess handle = observed total handle minus expected handle at baseline per journey.",
                    ],
                    style={"marginBottom": "0"},
                ),
            ],
            className="insight-card mb-3",
        )

        if HAS_SEABORN and not mystery.empty:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            sns.scatterplot(
                data=base,
                x="transfer_count",
                y="handle_total",
                hue="loop_flag",
                alpha=0.6,
                ax=ax1,
                palette={0: "#00A86B", 1: "#E81123"},
            )
            ax1.axhline(p95, linestyle="--", color="#111", linewidth=1)
            ax1.set_title("Transferred Journeys: Handle Time vs Transfer Count")
            ax1.set_xlabel("Transfer Count")
            ax1.set_ylabel("Total Handle Time")
            vis = seaborn_img_component(fig1, "mystery scatter", 540)
        else:
            vis = graph_component(scatter, 560)
        return html.Div(
            [
                explain,
                graph_component(uplift_fig, 480),
                graph_component(leak_fig, 620),
                vis,
                html.H5("Outlier Pathways"),
                tbl,
            ]
        )

    explorer_cols = [
        "master_contact_id",
        "contact_id",
        "CALL_START_WITH_COLLEAGUE",
        "INBOUND_OUTBOUND",
        "skill_name",
        "SKILL_COUNT",
        "HANDLETIME",
        "TALKTIME",
        "WRAPTIME",
        "LINE_OF_BUSINESS",
    ]
    table_df = filtered_c[explorer_cols].copy().sort_values(
        ["master_contact_id", "CALL_START_WITH_COLLEAGUE", "contact_id"]
    )
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in table_df.columns],
        data=table_df.to_dict("records"),
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header=DT_STYLE_HEADER,
        style_data=DT_STYLE_DATA,
        style_data_conditional=DT_STYLE_CONDITIONAL,
        style_cell={"textAlign": "left", "fontSize": "12px"},
    )


def create_sankey_from_paths(paths: list[list[str]], title: str) -> go.Figure:
    if not paths:
        fig = go.Figure()
        fig.add_annotation(
            text="No journey data available for this selection",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(title=title, height=620, autosize=True)
        return fig

    links = []
    for path in paths:
        for i in range(len(path) - 1):
            links.append((f"{path[i]} (Step {i+1})", f"{path[i+1]} (Step {i+2})"))

    link_counts = pd.Series(links).value_counts().reset_index()
    link_counts.columns = ["link", "count"]
    link_counts[["source", "target"]] = pd.DataFrame(link_counts["link"].tolist(), index=link_counts.index)

    all_nodes = list(set(link_counts["source"].tolist() + link_counts["target"].tolist()))
    node_dict = {n: i for i, n in enumerate(all_nodes)}
    colors = px.colors.qualitative.Set3
    node_colors = [colors[i % len(colors)] for i in range(len(all_nodes))]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=[node_dict[s] for s in link_counts["source"]],
                    target=[node_dict[t] for t in link_counts["target"]],
                    value=link_counts["count"].tolist(),
                    label=[f"{v} contacts" for v in link_counts["count"]],
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#2c3e50", family="Segoe UI")),
        font=dict(size=11, family="Segoe UI"),
        height=680,
        margin=dict(l=20, r=20, t=60, b=20),
        autosize=True,
    )
    return fig


@callback(
    Output("journey-analysis", "children"),
    Input("journey-queue-selector", "value"),
    Input("journey-depth-slider", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("direction-filter", "value"),
    Input("lob-filter", "value"),
    Input("queue-filter", "value"),
)
def update_journey_analysis(selected_skill, depth, start_date, end_date, direction, lobs, queues):
    if not selected_skill:
        return html.Div("Select a skill to view journey analysis.", className="alert alert-info")

    filtered_j = filter_journeys(start_date, end_date, direction, lobs or [], queues or [])
    filtered_c = filter_calls_by_journeys(filtered_j)
    q_ids = filtered_c[filtered_c["skill_name"] == selected_skill]["master_contact_id"].unique().tolist()
    if not q_ids:
        return html.Div("No journeys found for selected skill.", className="alert alert-warning")

    scope = (
        filtered_c[filtered_c["master_contact_id"].isin(q_ids)]
        .sort_values(["master_contact_id", "CALL_START_WITH_COLLEAGUE"])
        .copy()
    )

    forward_paths, backward_paths, complete_paths = [], [], []
    for cid in q_ids:
        j = scope[scope["master_contact_id"] == cid]["skill_name"].tolist()
        if not j:
            continue
        if selected_skill in j:
            idx = j.index(selected_skill)
            fpath = j[idx : idx + depth]
            bpath = j[max(0, idx - depth + 1) : idx + 1]
            if len(fpath) > 1:
                forward_paths.append(fpath)
            if len(bpath) > 1:
                backward_paths.append(bpath)
            complete_paths.append("→ ".join(j))

    if not complete_paths:
        return html.Div("No complete pathways available for this filter.", className="alert alert-warning")

    total_through = len(complete_paths)
    path_counts = pd.Series(complete_paths).value_counts().head(10).reset_index()
    path_counts.columns = ["Journey Path", "Journeys"]
    path_counts["% of Cases"] = (path_counts["Journeys"] / total_through * 100).round(1).astype(str) + "%"

    stats_cards = dbc.Row(
        [
            dbc.Col(
                [html.Div([html.H4("Cases Through Skill"), html.H2(f"{len(q_ids):,}")], className="kpi-card kpi-primary")],
                md=3,
            ),
            dbc.Col(
                [
                    html.Div(
                        [html.H4("Unique Forward Paths"), html.H2(f"{len(set(map(tuple, forward_paths)))}")],
                        className="kpi-card kpi-success",
                    )
                ],
                md=3,
            ),
            dbc.Col(
                [
                    html.Div(
                        [html.H4("Unique Backward Paths"), html.H2(f"{len(set(map(tuple, backward_paths)))}")],
                        className="kpi-card kpi-warning",
                    )
                ],
                md=3,
            ),
            dbc.Col(
                [
                    html.Div(
                        [html.H4("Avg Journey Length"), html.H2(f"{np.mean([p.count('→') + 1 for p in complete_paths]):.1f}")],
                        className="kpi-card kpi-info",
                    )
                ],
                md=3,
            ),
        ],
        className="mb-4",
    )

    path_table = dbc.Table.from_dataframe(
        path_counts, striped=True, bordered=True, hover=True, responsive=True, className="mt-2"
    )

    forward_sankey = create_sankey_from_paths(forward_paths, f"Forward Journey from {selected_skill}")
    backward_sankey = create_sankey_from_paths(backward_paths, f"Backward Journey to {selected_skill}")

    return html.Div(
        [
            stats_cards,
            html.Hr(className="divider"),
            html.H6(f"Forward View: Where do contacts go FROM {selected_skill}?", style={"fontWeight": "600"}),
            html.P("Paths contacts take after entering this skill.", className="text-muted"),
            graph_component(forward_sankey, 680),
            html.Hr(className="divider"),
            html.H6(f"Backward View: How do contacts arrive TO {selected_skill}?", style={"fontWeight": "600"}),
            html.P("Paths contacts took before reaching this skill.", className="text-muted"),
            graph_component(backward_sankey, 680),
            html.Hr(className="divider"),
            html.H6(f"Top 10 Complete Journey Paths Through {selected_skill}", style={"fontWeight": "600"}),
            html.P(
                f"All {total_through:,} journeys that touched this skill. Percentages sum to 100% across the listed paths.",
                className="text-muted",
            ),
            path_table,
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
