"""
Executive Case Routing Analytics Dashboard - Dash Version
8-Tab structure: Overview | Process | Cost & Effort | Hours Effect | Queue Intel | Journey | Data Explorer | ML Insights
"""

import os
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ALL, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ML imports
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# ==================================
# GENERATE SAMPLE DATA
# ==================================

def generate_sample_data():
    np.random.seed(42)
    sample_rows = []
    case_id = 1000

    queue_names = [
        'General Enquiry', 'Technical Support', 'Billing', 'Payments',
        'Account Management', 'Renewals', 'Cancellations', 'Escalations',
        'VIP Support', 'Customer Service', 'Complaints', 'Refunds'
    ]

    for _ in range(200):
        case_id += 1
        complexity = np.random.choice(['simple', 'medium', 'complex', 'very_complex'],
                                      p=[0.4, 0.3, 0.2, 0.1])
        inhours = np.random.choice([0, 1], p=[0.3, 0.7])

        if complexity == 'simple':
            num_queues = 1
            base_aht = np.random.uniform(30, 120) * 60  # seconds
            base_days = np.random.uniform(0, 1)
            base_messages = np.random.randint(1, 3)
        elif complexity == 'medium':
            num_queues = 2
            base_aht = np.random.uniform(80, 200) * 60  # seconds
            base_days = np.random.uniform(1, 3)
            base_messages = np.random.randint(2, 6)
        elif complexity == 'complex':
            num_queues = np.random.choice([3, 4])
            base_aht = np.random.uniform(150, 350) * 60  # seconds
            base_days = np.random.uniform(2, 7)
            base_messages = np.random.randint(4, 10)
        else:
            num_queues = np.random.choice([4, 5, 6])
            base_aht = np.random.uniform(250, 500) * 60  # seconds
            base_days = np.random.uniform(5, 15)
            base_messages = np.random.randint(6, 15)

        if inhours == 0:
            base_days *= 1.4
            base_aht *= 1.15
            base_messages = int(base_messages * 1.2)

        selected_queues = np.random.choice(queue_names, size=num_queues, replace=False).tolist()

        if np.random.random() < 0.1 and num_queues > 1:
            loop_queue = selected_queues[np.random.randint(0, len(selected_queues))]
            selected_queues.append(loop_queue)

        created_at = pd.Timestamp('2025-10-01') + pd.Timedelta(days=np.random.randint(0, 60))
        total_days = 0
        cumulative_aht = 0

        for queue_order, queue_name in enumerate(selected_queues, 1):
            if queue_order == len(selected_queues):
                days_in_queue = base_days * np.random.uniform(0.3, 0.7)
            else:
                days_in_queue = base_days * np.random.uniform(0.1, 0.4) / max(1, num_queues - 1)

            total_days += days_in_queue
            process_ts = created_at + pd.Timedelta(days=total_days)
            queue_aht = base_aht * (1 + (queue_order - 1) * 0.15) / num_queues
            cumulative_aht += queue_aht
            asrt = np.random.uniform(1, 5) * (1 + (queue_order - 1) * 0.3)
            close_datetime = created_at + pd.Timedelta(days=total_days) + pd.Timedelta(hours=np.random.randint(0, 24))
            hours_to_close = (close_datetime - created_at).total_seconds() / 3600
            interactions = base_messages + queue_order

            sample_rows.append({
                'CASE_ID': case_id,
                'QUEUE_ORDER': queue_order,
                'QUEUE_NEW': queue_name,
                'PROCESS_TIMESTAMP': process_ts,
                'DAYS_IN_QUEUE': round(days_in_queue, 2),
                'CREATED_AT': created_at,
                'CLOSE_DATE': close_datetime.strftime('%d/%m/%Y'),
                'CLOSE_TIME': close_datetime.strftime('%H:%M:%S.0'),
                'CLOSE_DATETIME': close_datetime,
                'HOURS_BETWEEN_CREATED_AND_CLOSE': round(hours_to_close, 2),
                'STATUS': 'closed',
                'TIMEFORASRT': round(asrt, 2),
                'NOOFINTERACTIONSFORASRT': interactions,
                'TOTALACTIVEAHT': round(cumulative_aht, 2),
                'AHT_TOTALCASE_HRS': round(cumulative_aht / 60, 2),
                'TOTALART': round(cumulative_aht * 1.1, 2),
                'INHOURS': inhours,
                'IN_OUTHRS_TYPE': 'InHours' if inhours == 1 else 'OutOfHours',
                'OUTOFHOURS': 1 - inhours,
                'MESSAGESRECEIVED_CUSTOMER': base_messages,
                'NOOFINTERACTIONS_INCFIRST': interactions + 1
            })

    return pd.DataFrame(sample_rows)


df_raw = generate_sample_data()

# ==================================
# DATA PREPARATION
# ==================================

def prepare_data(df):
    df = df.copy()
    df = df.sort_values(["CASE_ID", "QUEUE_ORDER"])

    case = (
        df.groupby("CASE_ID")
        .agg(
            transfers=("QUEUE_ORDER", lambda x: x.max() - 1),
            queues_touched=("QUEUE_ORDER", "max"),
            routing_days=("DAYS_IN_QUEUE", lambda x: x.iloc[:-1].sum() if len(x) > 1 else 0),
            final_queue_days=("DAYS_IN_QUEUE", lambda x: x.iloc[-1] if len(x) else 0),
            total_days_in_queue=("DAYS_IN_QUEUE", "sum"),
            total_active_aht=("TOTALACTIVEAHT", "max"),
            asrt=("TIMEFORASRT", "max"),
            messages=("MESSAGESRECEIVED_CUSTOMER", "max"),
            interactions=("NOOFINTERACTIONS_INCFIRST", "max"),
            inhours=("INHOURS", "max"),
            entry_queue=("QUEUE_NEW", "first"),
            final_queue=("QUEUE_NEW", "last"),
            close_hours=("HOURS_BETWEEN_CREATED_AND_CLOSE", "max"),
            created_at=("CREATED_AT", "first"),
            close_datetime=("CLOSE_DATETIME", "first"),
        )
        .reset_index()
    )

    loops = (
        df.groupby("CASE_ID")["QUEUE_NEW"]
        .apply(lambda x: x.duplicated().any())
        .astype(int)
        .reset_index(name="loop_flag")
    )
    case = case.merge(loops, on="CASE_ID", how="left")

    # Convert AHT from seconds to minutes (source data is in seconds)
    case['total_active_aht'] = case['total_active_aht'] / 60

    case['message_intensity'] = case['messages'] / (case['total_active_aht'].fillna(0) + 1)
    case['interaction_density'] = case['interactions'] / (case['total_active_aht'].fillna(0) + 1)
    case['ftr'] = (case['transfers'].fillna(0) == 0).astype(int)
    case['transfer_bin'] = pd.cut(case['transfers'].fillna(0),
                                  bins=[-0.1, 0, 1, 2, 100],
                                  labels=['0', '1', '2', '3+'])
    # Segment: Retail = starts with "HD RTL A" but NOT "HD RTL A PRT*"; Claims = everything else
    eq = case['entry_queue'].fillna('')
    case['segment'] = np.where(
        eq.str.startswith('HD RTL A') & ~eq.str.startswith('HD RTL A PRT'),
        'Retail', 'Claims'
    )
    return df, case


df_raw, case_df = prepare_data(df_raw)

# ==================================
# ML MODEL TRAINING
# ==================================

def build_ml_models(case_df, df_raw):
    """Train 3 ML models at startup. Compares RF vs GBM vs HistGBT for supervised tasks."""
    artifacts = {}
    df = case_df.copy()

    # Fill NaN on ML copy only — does NOT affect the shared case_df used by other tabs
    for col in ['transfers', 'inhours', 'messages', 'total_active_aht',
                'routing_days', 'loop_flag', 'message_intensity']:
        df[col] = df[col].fillna(0)
    df['inhours'] = df['inhours'].astype(int)

    df['day_of_week'] = pd.to_datetime(df['created_at'], errors='coerce').dt.dayofweek
    df['hour_of_day'] = pd.to_datetime(df['close_datetime'], errors='coerce').dt.hour
    df['day_of_week'] = df['day_of_week'].fillna(df['day_of_week'].median() if df['day_of_week'].notna().any() else 0).astype(int)
    df['hour_of_day'] = df['hour_of_day'].fillna(df['hour_of_day'].median() if df['hour_of_day'].notna().any() else 12).astype(int)

    feature_cols_base = ['inhours', 'day_of_week', 'hour_of_day']
    entry_dummies = pd.get_dummies(df['entry_queue'], prefix='eq')
    X = pd.concat([df[feature_cols_base], entry_dummies], axis=1).astype(float)
    # Fill any remaining NaN (ensures GradientBoostingClassifier compatibility)
    X = X.fillna(0)

    # Drop rows with NaN in target columns for training, predict on all
    valid_mask = df['ftr'].notna() & df['final_queue'].notna() & df['entry_queue'].notna()
    X_valid_all = X[valid_mask]
    df_valid_all = df[valid_mask]

    # Sample for training speed — train on up to 3000 rows, predict on ALL rows
    ML_SAMPLE_SIZE = 3000
    if len(df_valid_all) > ML_SAMPLE_SIZE:
        sample_idx = df_valid_all.sample(n=ML_SAMPLE_SIZE, random_state=42).index
        X_valid = X_valid_all.loc[sample_idx]
        df_valid = df_valid_all.loc[sample_idx]
    else:
        X_valid = X_valid_all
        df_valid = df_valid_all

    # Use fewer folds if dataset is small or has rare classes
    min_class_count = min(df_valid['ftr'].value_counts().min(),
                          df_valid['final_queue'].value_counts().min())
    n_splits = min(5, max(2, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # MODEL 1: Transfer Risk (Binary)
    y_transfer = (df_valid['ftr'] == 0).astype(int)

    candidates_m1 = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=10,
            class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=80, max_depth=4, min_samples_leaf=10,
            learning_rate=0.1, random_state=42),
        'Hist Gradient Boost': HistGradientBoostingClassifier(
            max_iter=80, max_depth=4, min_samples_leaf=10,
            learning_rate=0.1, class_weight='balanced', random_state=42),
    }

    m1_scores = {}
    for name, model in candidates_m1.items():
        scores = cross_val_score(model, X_valid, y_transfer, cv=cv, scoring='roc_auc')
        m1_scores[name] = {'mean': scores.mean(), 'std': scores.std()}

    best_m1_name = max(m1_scores, key=lambda k: m1_scores[k]['mean'])
    best_m1 = candidates_m1[best_m1_name]
    best_m1.fit(X_valid, y_transfer)
    df['transfer_risk'] = (best_m1.predict_proba(X)[:, 1] * 100).round(1)

    if hasattr(best_m1, 'feature_importances_'):
        m1_importances = best_m1.feature_importances_
    else:
        perm = permutation_importance(best_m1, X_valid, y_transfer, n_repeats=10, random_state=42)
        m1_importances = perm.importances_mean

    artifacts['model1'] = {
        'model': best_m1, 'best_name': best_m1_name, 'all_scores': m1_scores,
        'feature_names': list(X.columns), 'importances': m1_importances,
        'cv_auc_mean': m1_scores[best_m1_name]['mean'],
        'cv_auc_std': m1_scores[best_m1_name]['std'],
    }

    # MODEL 2: Optimal First-Queue Recommendation (Multiclass)
    y_queue = df_valid['final_queue']

    candidates_m2 = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=5,
            class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=80, max_depth=5, min_samples_leaf=5,
            learning_rate=0.1, random_state=42),
        'Hist Gradient Boost': HistGradientBoostingClassifier(
            max_iter=80, max_depth=5, min_samples_leaf=5,
            learning_rate=0.1, class_weight='balanced', random_state=42),
    }

    m2_scores = {}
    for name, model in candidates_m2.items():
        scores = cross_val_score(model, X_valid, y_queue, cv=cv, scoring='accuracy')
        m2_scores[name] = {'mean': scores.mean(), 'std': scores.std()}

    best_m2_name = max(m2_scores, key=lambda k: m2_scores[k]['mean'])
    best_m2 = candidates_m2[best_m2_name]
    best_m2.fit(X_valid, y_queue)
    df['recommended_queue'] = best_m2.predict(X)
    df['queue_match'] = (df['recommended_queue'] == df['final_queue']).astype(int)

    if hasattr(best_m2, 'feature_importances_'):
        m2_importances = best_m2.feature_importances_
    else:
        perm = permutation_importance(best_m2, X_valid, y_queue, n_repeats=10, random_state=42)
        m2_importances = perm.importances_mean

    artifacts['model2'] = {
        'model': best_m2, 'best_name': best_m2_name, 'all_scores': m2_scores,
        'feature_names': list(X.columns), 'importances': m2_importances,
        'cv_acc_mean': m2_scores[best_m2_name]['mean'],
        'cv_acc_std': m2_scores[best_m2_name]['std'],
    }

    # MODEL 3: Journey Clustering (Unsupervised)
    queue_counts = df_raw.groupby(['CASE_ID', 'QUEUE_NEW']).size().unstack(fill_value=0)
    queue_counts.columns = ['qc_' + c for c in queue_counts.columns]

    cluster_features = ['transfers', 'routing_days', 'total_active_aht',
                        'messages', 'loop_flag', 'inhours', 'message_intensity']
    X_cluster_base = df.set_index('CASE_ID')[cluster_features].fillna(0)
    X_cluster_all = X_cluster_base.join(queue_counts, how='left').fillna(0)

    # Sample for clustering training (silhouette + KMeans fit)
    if len(X_cluster_all) > ML_SAMPLE_SIZE:
        X_cluster_sample = X_cluster_all.sample(n=ML_SAMPLE_SIZE, random_state=42)
    else:
        X_cluster_sample = X_cluster_all

    scaler = StandardScaler()
    X_scaled_sample = scaler.fit_transform(X_cluster_sample)

    best_k, best_sil = 4, -1
    for k in [3, 4, 5, 6]:
        km_temp = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_temp = km_temp.fit_predict(X_scaled_sample)
        sil = silhouette_score(X_scaled_sample, labels_temp)
        if sil > best_sil:
            best_k, best_sil = k, sil

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    km.fit(X_scaled_sample)

    # Predict on ALL rows, PCA on ALL rows
    X_scaled_all = scaler.transform(X_cluster_all)
    cluster_labels = km.predict(X_scaled_all)

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled_all)

    df['journey_cluster'] = cluster_labels

    cluster_profiles = df.groupby('journey_cluster').agg(
        avg_transfers=('transfers', 'mean'), avg_routing=('routing_days', 'mean'),
        avg_aht=('total_active_aht', 'mean'), avg_messages=('messages', 'mean'),
        loop_rate=('loop_flag', 'mean'), count=('CASE_ID', 'count'),
    )

    sorted_clusters = cluster_profiles.sort_values('avg_transfers')
    name_map = {}
    for i, (idx, row) in enumerate(sorted_clusters.iterrows()):
        if row['loop_rate'] > 0.25 and row['avg_transfers'] > 1:
            name_map[idx] = 'Ping-Pong Loop'
        elif row['avg_transfers'] < 0.5:
            name_map[idx] = 'Quick Resolve'
        elif row['avg_transfers'] >= 3:
            name_map[idx] = 'Complex Multi-Queue'
        elif row['avg_aht'] > cluster_profiles['avg_aht'].median() * 1.3:
            name_map[idx] = 'High-Effort Escalation'
        else:
            name_map[idx] = 'Standard Escalation'

    seen = {}
    for idx in sorted(name_map.keys()):
        base = name_map[idx]
        if base in seen:
            name_map[idx] = f"{base} (Group {idx + 1})"
            if seen[base] == 1:
                first_idx = [k for k, v in name_map.items() if v == base][0]
                name_map[first_idx] = f"{base} (Group {first_idx + 1})"
            seen[base] += 1
        else:
            seen[base] = 1

    df['cluster_name'] = df['journey_cluster'].map(name_map)

    artifacts['model3'] = {
        'model': km, 'scaler': scaler, 'pca': pca, 'pca_coords': pca_coords,
        'silhouette': best_sil, 'best_k': best_k,
        'cluster_profiles': cluster_profiles, 'name_map': name_map,
        'feature_names': list(X_cluster_all.columns),
    }

    # Only return ML-specific columns to add back — don't overwrite original data
    ml_cols = ['day_of_week', 'hour_of_day', 'transfer_risk', 'recommended_queue',
               'queue_match', 'journey_cluster', 'cluster_name']
    return df[['CASE_ID'] + ml_cols], artifacts


_ml_result, ml_artifacts = build_ml_models(case_df, df_raw)
# Merge ML columns into case_df without overwriting original values
for col in _ml_result.columns:
    if col != 'CASE_ID':
        case_df[col] = _ml_result[col].values
del _ml_result

min_date = case_df['created_at'].min().date()
max_date = case_df['created_at'].max().date()

# ==================================
# APP INITIALISATION
# ==================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.title = "Messenger Transfer Analytics - Hastings Direct"

POWERBI_COLORS = {
    'primary': '#00BCF2',
    'secondary': '#742774',
    'success': '#00A86B',
    'warning': '#FFB900',
    'danger': '#E81123',
    'dark': '#252423',
    'light': '#F3F2F1'
}

CHART_COLORS = ['#00BCF2', '#742774', '#FFB900', '#E81123', '#00A86B',
                '#8764B8', '#F2C80F', '#0078D4', '#107C10', '#C50F1F']

# ==================================
# HELPERS
# ==================================

def create_filter_section(tab_id):
    """Power BI slicer-style filter panel — one card per filter."""
    return html.Div([
        html.Div([
            html.Span("REPORT FILTERS", style={
                'fontSize': '0.7rem', 'fontWeight': '700',
                'color': '#777', 'letterSpacing': '1.3px'
            })
        ], style={'paddingBottom': '10px', 'marginBottom': '12px',
                  'borderBottom': '1px solid #E0E0E0'}),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("DATE RANGE", style={
                            'fontSize': '0.7rem', 'fontWeight': '700',
                            'color': '#444', 'letterSpacing': '0.5px'
                        })
                    ], className="slicer-header"),
                    html.Div([
                        dcc.DatePickerRange(
                            id=f'{tab_id}-date-filter',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date,
                            display_format='DD/MM/YYYY',
                            style={'fontSize': '0.82rem'}
                        ),
                    ], className="slicer-body")
                ], className="slicer-card")
            ], md=3),

            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("ENTRY QUEUE", style={
                            'fontSize': '0.7rem', 'fontWeight': '700',
                            'color': '#444', 'letterSpacing': '0.5px'
                        })
                    ], className="slicer-header"),
                    html.Div([
                        dcc.Dropdown(
                            id=f'{tab_id}-queue-filter',
                            options=[{'label': q, 'value': q}
                                     for q in sorted(case_df.entry_queue.dropna().unique())],
                            value=[],
                            multi=True,
                            placeholder="All queues (select to filter)",
                            style={'fontSize': '0.82rem'}
                        ),
                    ], className="slicer-body")
                ], className="slicer-card")
            ], md=3),

            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("HOURS TYPE", style={
                            'fontSize': '0.7rem', 'fontWeight': '700',
                            'color': '#444', 'letterSpacing': '0.5px'
                        })
                    ], className="slicer-header"),
                    html.Div([
                        dcc.Dropdown(
                            id=f'{tab_id}-hours-filter',
                            options=[
                                {'label': 'In Hours', 'value': 1},
                                {'label': 'Out of Hours', 'value': 0}
                            ],
                            value=[0, 1],
                            multi=True,
                            placeholder="Select hours type...",
                            style={'fontSize': '0.82rem'}
                        ),
                    ], className="slicer-body")
                ], className="slicer-card")
            ], md=3),

            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("SEGMENT", style={
                            'fontSize': '0.7rem', 'fontWeight': '700',
                            'color': '#444', 'letterSpacing': '0.5px'
                        })
                    ], className="slicer-header"),
                    html.Div([
                        dcc.Dropdown(
                            id=f'{tab_id}-segment-filter',
                            options=[
                                {'label': 'Retail', 'value': 'Retail'},
                                {'label': 'Claims', 'value': 'Claims'}
                            ],
                            value=['Retail', 'Claims'],
                            multi=True,
                            placeholder="Select segment...",
                            style={'fontSize': '0.82rem'}
                        ),
                    ], className="slicer-body")
                ], className="slicer-card")
            ], md=3),
        ], className="g-3"),

    ], className="filter-panel mb-4")


def guide_statement(children):
    """Render a business guide statement banner at the top of a tab.
    Accepts a string or list of html elements (use html.Strong for bold)."""
    return html.Div(
        html.P(children, style={'margin': 0, 'fontSize': '0.88rem', 'color': '#444',
                                'lineHeight': '1.6', 'fontStyle': 'italic'}),
        style={'background': '#F3F2F1', 'borderLeft': '3px solid #0078D4',
               'borderRadius': '0 6px 6px 0', 'padding': '0.8rem 1.2rem',
               'marginBottom': '1.2rem'}
    )


def filter_data(case_data, start_date, end_date, queues, hours, segments=None):
    if not hours:
        return pd.DataFrame()
    mask = (
        (case_data.created_at.dt.date >= pd.to_datetime(start_date).date()) &
        (case_data.created_at.dt.date <= pd.to_datetime(end_date).date()) &
        (case_data.inhours.isin(hours))
    )
    # Empty queue selection = all queues
    if queues:
        mask = mask & case_data.entry_queue.isin(queues)
    if segments:
        mask = mask & case_data.segment.isin(segments)
    return case_data[mask]


# ==================================
# LAYOUT — 6 TABS
# ==================================

app.layout = dbc.Container([

    # ── Global header with logo ──────────────────────────────────────────────
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Img(
                    src='/assets/hastings_logo.svg',
                    style={'height': '44px', 'objectFit': 'contain'},
                    title='Hastings Direct'
                ),
            ], xs='auto', className="d-flex align-items-center"),

            dbc.Col([
                html.Div([
                    html.H1("Messenger Transfer Analytics",
                            style={'color': POWERBI_COLORS['primary'], 'fontWeight': '700',
                                   'fontSize': '1.8rem', 'marginBottom': '0.15rem',
                                   'lineHeight': '1.2'}),
                    html.P("Understanding the cost of transfers across Messenger cases",
                           style={'color': '#605E5C', 'fontSize': '0.88rem', 'marginBottom': '0'}),
                ])
            ], className="d-flex align-items-center"),

            dbc.Col([
                html.Div([
                    html.Div("INTERNAL USE ONLY", style={
                        'fontSize': '0.65rem', 'fontWeight': '700', 'letterSpacing': '1.2px',
                        'color': '#A0A0A0', 'textTransform': 'uppercase', 'textAlign': 'right',
                    }),
                    html.Div("Customer Operations", style={
                        'fontSize': '0.75rem', 'color': '#888', 'textAlign': 'right',
                    }),
                ])
            ], xs='auto', className="d-flex align-items-center ms-auto"),
        ], align="center", className="g-3"),
        html.Hr(className="divider", style={'marginTop': '0.9rem', 'marginBottom': '0'}),
    ], className="mb-3", style={'paddingTop': '0.5rem'}),

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Overview & Definitions', value='tab-1'),
        dcc.Tab(label='Process & Routing',       value='tab-2'),
        dcc.Tab(label='Cost & Effort Impact',    value='tab-3'),
        dcc.Tab(label='Hours & Transfer Effect',  value='tab-4'),
        dcc.Tab(label='Queue Intelligence',       value='tab-5'),
        dcc.Tab(label='Journey Pathways',         value='tab-6'),
        dcc.Tab(label='Data Explorer',            value='tab-7'),
        dcc.Tab(label='ML Insights',              value='tab-8'),
    ]),

    html.Div(id='tabs-content', className="mt-4")

], fluid=True)


# ==================================
# RENDER TABS
# ==================================

@callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return build_landing_page()
    elif tab == 'tab-2':
        return html.Div([create_filter_section('process'), html.Div(id='process-content')])
    elif tab == 'tab-3':
        return html.Div([create_filter_section('impact'),  html.Div(id='impact-content')])
    elif tab == 'tab-4':
        return html.Div([create_filter_section('hours'),   html.Div(id='hours-content')])
    elif tab == 'tab-5':
        return html.Div([create_filter_section('qi'),      html.Div(id='qi-content')])
    elif tab == 'tab-6':
        return html.Div([create_filter_section('journey'), html.Div(id='journey-content')])
    elif tab == 'tab-7':
        return html.Div([create_filter_section('explorer'), html.Div(id='explorer-content')])
    elif tab == 'tab-8':
        return build_ml_insights_tab()


# ==================================
# TAB 1: LANDING / DEFINITION PAGE
# ==================================

def build_landing_page():
    """
    Overview & Definitions landing page — professional, clean, Messenger-branded.
    Uses full dataset for live health stats. No filters needed.
    """

    # ── Live health stats from full dataset ──────────────────────────────────
    total_cases       = len(case_df)
    ftr_rate          = case_df.ftr.mean() * 100
    avg_transfers     = case_df.transfers.mean()
    multi_xfer_pct    = (case_df.transfers >= 2).mean() * 100
    ooh_pct           = (case_df.inhours == 0).mean() * 100
    median_aht        = case_df.total_active_aht.median()
    routing_waste_pct = (case_df.routing_days.sum() /
                         max(case_df.total_days_in_queue.sum(), 0.001)) * 100
    loop_pct          = case_df.loop_flag.mean() * 100

    # ── Shared component helpers ─────────────────────────────────────────────
    CARD = {
        'background': 'white', 'borderRadius': '8px', 'height': '100%',
        'boxShadow': '0 1.6px 3.6px 0 rgba(0,0,0,.132)',
        'padding': '1.2rem 1.4rem', 'borderTop': '3px solid',
    }
    DEF_ITEM = {
        'padding': '0.85rem 1rem', 'borderRadius': '6px',
        'marginBottom': '0.6rem', 'borderLeft': '3px solid', 'background': '#FAFAFA',
    }

    def kpi_stat(value, label, color):
        return html.Div([
            html.Div(value, style={
                'fontSize': '1.65rem', 'fontWeight': '700', 'color': color,
                'lineHeight': '1.1', 'marginBottom': '0.2rem',
            }),
            html.Div(label, style={
                'fontSize': '0.7rem', 'fontWeight': '600', 'color': '#888',
                'textTransform': 'uppercase', 'letterSpacing': '0.6px',
            }),
        ], style={
            'textAlign': 'center', 'padding': '0.9rem 0.5rem',
            'borderRight': '1px solid #EDEBE9',
        })

    def section_card(icon, title, color, questions, tab_hint):
        header_children = []
        if icon:
            header_children.append(html.Span(icon, style={'fontSize': '1.3rem', 'marginRight': '0.5rem'}))
        header_children.append(html.Span(title, style={'fontSize': '0.95rem', 'fontWeight': '700',
                                        'color': '#201F1E', 'verticalAlign': 'middle'}))
        return html.Div([
            html.Div(header_children,
                     style={'marginBottom': '0.7rem', 'display': 'flex', 'alignItems': 'center'}),
            html.Ul([
                html.Li(q, style={'fontSize': '0.82rem', 'color': '#605E5C',
                                  'marginBottom': '0.25rem'}) for q in questions
            ], style={'paddingLeft': '1rem', 'margin': '0 0 0.8rem 0'}),
            html.Div(tab_hint, style={
                'fontSize': '0.7rem', 'fontWeight': '700', 'color': color,
                'textTransform': 'uppercase', 'letterSpacing': '0.5px',
                'borderTop': f'1px solid {color}25', 'paddingTop': '0.55rem',
            })
        ], style={**CARD, 'borderTopColor': color})

    def def_item(color, term, definition):
        return html.Div([
            html.Div(term, style={'fontWeight': '700', 'fontSize': '0.85rem',
                                  'color': '#201F1E', 'marginBottom': '0.2rem'}),
            html.Div(definition, style={'fontSize': '0.81rem', 'color': '#605E5C',
                                        'lineHeight': '1.55'}),
        ], style={**DEF_ITEM, 'borderLeftColor': color})

    # ── PURPOSE STATEMENT ────────────────────────────────────────────────────
    purpose = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("ABOUT THIS REPORT", style={
                        'fontSize': '0.65rem', 'fontWeight': '700', 'letterSpacing': '1.5px',
                        'color': '#00BCF2', 'marginBottom': '0.5rem', 'textTransform': 'uppercase',
                    }),
                    html.H4("Understanding the True Cost of Messenger Transfers",
                            style={'fontWeight': '700', 'color': '#201F1E',
                                   'fontSize': '1.2rem', 'marginBottom': '0.7rem'}),
                    html.P([
                        "When a Messenger case is transferred between queues, three costs accumulate silently: "
                        "the time a customer waits while their case is in transit (",
                        html.Strong("Waiting"), "), the extra agent effort required each time a new advisor "
                        "picks up the case (",
                        html.Strong("Working"), "), and the frustration a customer expresses through repeated "
                        "messages and chasing (",
                        html.Strong("Friction"), "). This report isolates each dimension so the business "
                        "can act on the right lever."
                    ], style={'fontSize': '0.88rem', 'color': '#605E5C',
                              'lineHeight': '1.65', 'marginBottom': '0'}),
                ], style={'borderLeft': '4px solid #00BCF2', 'paddingLeft': '1.2rem'})
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Div("DATA SNAPSHOT", style={
                        'fontSize': '0.65rem', 'fontWeight': '700', 'letterSpacing': '1.2px',
                        'color': '#888', 'marginBottom': '0.8rem', 'textTransform': 'uppercase',
                        'textAlign': 'center',
                    }),
                    dbc.Row([
                        dbc.Col([kpi_stat(f"{total_cases:,}", "Messenger Cases", '#00BCF2')], xs=6),
                        dbc.Col([kpi_stat(f"{ftr_rate:.0f}%", "Direct Resolution", '#00A86B')], xs=6,
                                style={'borderRight': 'none'}),
                    ], className="g-0 mb-2"),
                    dbc.Row([
                        dbc.Col([kpi_stat(f"{avg_transfers:.2f}", "Avg Transfers", '#FFB900')], xs=6),
                        dbc.Col([kpi_stat(f"{median_aht:.0f}m", "Median AHT", '#742774')], xs=6,
                                style={'borderRight': 'none'}),
                    ], className="g-0 mb-2"),
                    dbc.Row([
                        dbc.Col([kpi_stat(f"{routing_waste_pct:.0f}%", "Routing Waste", '#E81123')], xs=6),
                        dbc.Col([kpi_stat(f"{ooh_pct:.0f}%", "Out-of-Hours", '#0078D4')], xs=6,
                                style={'borderRight': 'none'}),
                    ], className="g-0"),
                ], style={
                    'background': '#F8F8F8', 'borderRadius': '6px',
                    'padding': '0.9rem 0.5rem', 'border': '1px solid #EDEBE9',
                })
            ], md=4, className="align-self-center"),
        ], className="g-4"),
    ], style={
        'background': 'white', 'borderRadius': '8px', 'padding': '1.5rem 1.8rem',
        'boxShadow': '0 1.6px 3.6px 0 rgba(0,0,0,.132)', 'marginBottom': '1.25rem',
    })

    # ── DATA SIGNALS CALLOUT ─────────────────────────────────────────────────
    alert_data = []
    if multi_xfer_pct > 20:
        alert_data.append(
            f"{multi_xfer_pct:.0f}% of Messenger cases require 2+ transfers — see Cost & Effort Impact (Tab 3) and Queue Intelligence (Tab 5)."
        )
    if ooh_pct > 25:
        alert_data.append(
            f"{ooh_pct:.0f}% of cases are created out-of-hours — see Hours & Transfer Effect (Tab 4) for the compounding impact."
        )
    if loop_pct > 5:
        alert_data.append(
            f"{loop_pct:.0f}% of cases loop back to a previously-visited queue — see Journey Pathways (Tab 6)."
        )

    alert_block = html.Div()
    if alert_data:
        alert_block = html.Div([
            html.Div([
                html.Strong("Signals Worth Investigating "),
                html.Span("Based on the full dataset loaded.",
                          style={'fontSize': '0.8rem', 'color': '#888', 'fontWeight': '400'}),
            ], style={'marginBottom': '0.5rem', 'fontSize': '0.88rem'}),
            html.Ul([html.Li(a, style={'fontSize': '0.83rem', 'marginBottom': '0.25rem'})
                     for a in alert_data],
                    style={'marginBottom': '0', 'paddingLeft': '1.1rem'}),
        ], className="insight-card mb-3")

    # ── SECTION GUIDE ────────────────────────────────────────────────────────
    section_guide = html.Div([
        html.Div([
            html.H5("Report Sections", style={
                'fontWeight': '700', 'color': '#201F1E', 'marginBottom': '0.2rem', 'fontSize': '1rem',
            }),
            html.P("Select any tab above to open that section.",
                   style={'color': '#888', 'fontSize': '0.82rem', 'marginBottom': '1rem'}),
        ]),
        dbc.Row([
            dbc.Col([section_card(
                "", "Process & Routing", '#0078D4',
                ["Which queues cause the most delay?",
                 "What % of time is wasted in transit vs. active resolution?",
                 "Where do loop-backs and re-routing occur?"],
                "Tab 2 — Queue delay breakdown and routing efficiency"
            )], md=4, className="mb-3"),
            dbc.Col([section_card(
                "", "Cost & Effort Impact", '#00A86B',
                ["How much does each transfer inflate handle time?",
                 "At what point does transfer cost become unacceptable?",
                 "How does customer messaging scale with routing friction?"],
                "Tab 3 — AHT inflation curves and effort escalation index"
            )], md=4, className="mb-3"),
            dbc.Col([section_card(
                "", "Hours & Transfer Effect", '#E81123',
                ["Do out-of-hours Messenger cases attract more transfers?",
                 "What is the compounding cost of OOH + multiple transfers?",
                 "Which hour × transfer combination drives the highest AHT?"],
                "Tab 4 — OOH impact and AHT heatmap analysis"
            )], md=4, className="mb-3"),
            dbc.Col([section_card(
                "", "Queue Intelligence", '#742774',
                ["For a given queue: who sends cases in, and where do they go next?",
                 "Which queues have the worst resolution metrics?",
                 "What does each queue's transfer pattern look like?"],
                "Tab 5 — Queue deep-dive with inbound and outbound flow"
            )], md=6, className="mb-3"),
            dbc.Col([section_card(
                "", "Journey Pathways", '#FFB900',
                ["What are the most common multi-queue journeys for Messenger cases?",
                 "What % of cases follow each routing path?",
                 "Which paths carry the most volume and the highest cost?"],
                "Tab 6 — End-to-end journey mapping and path frequency"
            )], md=6, className="mb-3"),
        ]),
    ], style={
        'background': 'white', 'borderRadius': '8px', 'padding': '1.4rem 1.6rem',
        'boxShadow': '0 1.6px 3.6px 0 rgba(0,0,0,.132)', 'marginBottom': '1.25rem',
    })

    # ── THREE DIMENSIONS ─────────────────────────────────────────────────────
    def dim_pill(icon, color, label, body):
        pill_header = []
        if icon:
            pill_header.append(html.Span(icon, style={'fontSize': '1.2rem', 'marginRight': '0.5rem'}))
        pill_header.append(html.Span(label, style={'fontWeight': '700', 'color': color,
                                        'fontSize': '0.88rem', 'textTransform': 'uppercase',
                                        'letterSpacing': '0.5px'}))
        return html.Div([
            html.Div(pill_header,
                     style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '0.5rem'}),
            html.P(body, style={'fontSize': '0.82rem', 'color': '#605E5C',
                                'lineHeight': '1.55', 'margin': '0'}),
        ], style={
            'borderLeft': f'3px solid {color}', 'paddingLeft': '1rem',
            'background': f'{color}08', 'borderRadius': '0 6px 6px 0',
            'padding': '0.9rem 1rem', 'height': '100%',
        })

    three_dims = html.Div([
        html.Div([
            html.H5("The Three Dimensions of Transfer Cost", style={
                'fontWeight': '700', 'color': '#201F1E', 'marginBottom': '0.2rem', 'fontSize': '1rem',
            }),
            html.P("Every metric in this report maps to one of these three categories.",
                   style={'color': '#888', 'fontSize': '0.82rem', 'marginBottom': '1rem'}),
        ]),
        dbc.Row([
            dbc.Col([dim_pill(
                "", "#E81123", "Waiting",
                "Calendar time lost while a Messenger case travels between queues. "
                "Customers experience this as slow resolution. Measured as Routing Days — "
                "days the case spent in transit before reaching the queue that resolved it."
            )], md=4, className="mb-3"),
            dbc.Col([dim_pill(
                "", "#0078D4", "Working",
                "Productive agent time spent on the case. Each transfer inflates this because "
                "a new advisor must re-read the conversation, re-engage the customer, and "
                "re-process what was already done. Measured as AHT (minutes)."
            )], md=4, className="mb-3"),
            dbc.Col([dim_pill(
                "", "#742774", "Friction",
                "Customer effort generated by poor routing. When Messenger cases bounce between "
                "queues, customers send more messages, ask the same questions again, and are "
                "more likely to escalate. Measured as customer message count."
            )], md=4, className="mb-3"),
        ]),
    ], style={
        'background': 'white', 'borderRadius': '8px', 'padding': '1.4rem 1.6rem',
        'boxShadow': '0 1.6px 3.6px 0 rgba(0,0,0,.132)', 'marginBottom': '1.25rem',
    })

    # ── KEY DEFINITIONS ──────────────────────────────────────────────────────
    defs = html.Div([
        html.Div([
            html.H5("Key Definitions", style={
                'fontWeight': '700', 'color': '#201F1E', 'marginBottom': '0.2rem', 'fontSize': '1rem',
            }),
            html.P("Plain-language definitions for every metric used across this report.",
                   style={'color': '#888', 'fontSize': '0.82rem', 'marginBottom': '1rem'}),
        ]),
        dbc.Row([
            dbc.Col([
                def_item('#E81123', 'Transfer',
                         'A Messenger case moving from one queue or team to another before resolution. '
                         'Each transfer is a handoff — the new advisor starts from scratch.'),
                def_item('#00BCF2', 'Direct Resolution Rate (DRR)',
                         'The % of Messenger cases resolved without any transfer — handled entirely '
                         'in the first queue they entered. Also called First-Touch Resolution (FTR). '
                         'Definition: cases where number of transfers = 0.'),
                def_item('#FFB900', 'Routing Days',
                         'Calendar time a case spends moving between queues before it reaches the '
                         'queue that resolves it. Pure delay — no value added. The "waiting" cost.'),
                def_item('#0078D4', 'AHT (Average Handle Time)',
                         'Total active agent time spent on a Messenger case, in minutes. '
                         'Inflates with each transfer as new advisors re-read and re-process.'),
            ], md=6),
            dbc.Col([
                def_item('#742774', 'Customer Messages',
                         'The number of messages a customer sends on the Messenger case. '
                         'Higher transfer counts drive more customer messages — customers chase '
                         'updates, repeat their issue, and push back when routing fails them.'),
                def_item('#00A86B', 'Out-of-Hours (OOH)',
                         'Messenger cases created outside standard business hours. OOH cases sit '
                         'in queues until teams come online, compounding both waiting time and '
                         'the likelihood of additional transfers.'),
                def_item('#E81123', 'Loop / Re-queue',
                         'When a Messenger case is routed back to a queue it has already visited. '
                         'Signals unclear ownership or skill mismatches in routing rules. '
                         'Every loop adds at least one transfer and one round of re-reading.'),
                def_item('#0078D4', 'Journey / Pathway',
                         'The ordered sequence of queues a Messenger case passes through. '
                         'E.g. General Enquiry → Technical Support → Escalations. '
                         'The report maps the most frequent paths to surface routing patterns.'),
            ], md=6),
        ]),
    ], style={
        'background': 'white', 'borderRadius': '8px', 'padding': '1.4rem 1.6rem',
        'boxShadow': '0 1.6px 3.6px 0 rgba(0,0,0,.132)',
    })

    return html.Div([
        purpose,
        alert_block,
        section_guide,
        three_dims,
        defs,
    ])


# ==================================
# TAB 2: PROCESS & ROUTING
# ==================================

@callback(
    Output('process-content', 'children'),
    [Input('process-date-filter', 'start_date'), Input('process-date-filter', 'end_date'),
     Input('process-queue-filter', 'value'),    Input('process-hours-filter', 'value'),
     Input('process-segment-filter', 'value')]
)
def update_process_tab(start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    filtered_cases = filtered.CASE_ID.unique()
    df_filtered = df_raw[df_raw.CASE_ID.isin(filtered_cases)]

    # Intermediary queues only: exclude first (entry) and last (resolution) queue per case
    # These are the "bottleneck" queues where cases sit waiting during transfers
    max_order = df_filtered.groupby('CASE_ID')['QUEUE_ORDER'].transform('max')
    intermediary = df_filtered[
        (df_filtered['QUEUE_ORDER'] > 1) &
        (df_filtered['QUEUE_ORDER'] < max_order)
    ]

    if len(intermediary) > 0:
        queue_impact = (
            intermediary.groupby("QUEUE_NEW")
            .agg(total_delay_days=("DAYS_IN_QUEUE", "sum"),
                 median_days=("DAYS_IN_QUEUE", "median"),
                 volume=("CASE_ID", "nunique"))
            .sort_values("total_delay_days", ascending=False)
            .head(10).reset_index()
        )
    else:
        # Fallback if no intermediary queues (all cases are 0-1 transfers)
        queue_impact = (
            df_filtered.groupby("QUEUE_NEW")
            .agg(total_delay_days=("DAYS_IN_QUEUE", "sum"),
                 median_days=("DAYS_IN_QUEUE", "median"),
                 volume=("CASE_ID", "nunique"))
            .sort_values("total_delay_days", ascending=False)
            .head(10).reset_index()
        )

    queue_impact['cumulative_pct'] = (queue_impact['total_delay_days'].cumsum() /
                                      queue_impact['total_delay_days'].sum() * 100)

    pareto_fig = make_subplots(specs=[[{"secondary_y": True}]])
    pareto_fig.add_trace(
        go.Bar(x=queue_impact['QUEUE_NEW'], y=queue_impact['total_delay_days'],
               name="Total Delay Days", marker_color=POWERBI_COLORS['danger']),
        secondary_y=False)
    pareto_fig.add_trace(
        go.Scatter(x=queue_impact['QUEUE_NEW'], y=queue_impact['cumulative_pct'],
                   name="Cumulative %", mode='lines+markers',
                   marker_color=POWERBI_COLORS['dark'], line=dict(width=2)),
        secondary_y=True)
    pareto_fig.update_xaxes(title_text="Intermediary Queue")
    pareto_fig.update_yaxes(title_text="Total Delay Days", secondary_y=False)
    pareto_fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
    pareto_title = ("Top 10 Intermediary Bottleneck Queues (80/20 Pareto)"
                    if len(intermediary) > 0 else "Top 10 Bottleneck Queues (80/20 Pareto)")
    pareto_fig.update_layout(
        title=dict(text=pareto_title,
                   font=dict(size=13, color='#201F1E', family='Segoe UI')),
        width=550, height=480, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI'), yaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        margin=dict(l=50, r=20, t=60, b=80)
    )

    entry_perf = (
        filtered.groupby("entry_queue")
        .agg(cases=("CASE_ID", "count"), ftr_rate=("ftr", "mean"))
        .sort_values("ftr_rate", ascending=True).head(10).reset_index()
    )
    entry_fig = go.Figure()
    entry_fig.add_trace(go.Bar(y=entry_perf['entry_queue'], x=entry_perf['ftr_rate'] * 100,
                                orientation='h', name='FTR %', marker_color=POWERBI_COLORS['success']))
    entry_fig.add_trace(go.Bar(y=entry_perf['entry_queue'], x=(1 - entry_perf['ftr_rate']) * 100,
                                orientation='h', name='Transfer %', marker_color=POWERBI_COLORS['danger']))
    entry_fig.update_layout(
        title="Entry Queue FTR Performance (worst → best)",
        barmode='stack', xaxis_title="% of Cases",
        width=550, height=480, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI'), xaxis=dict(showgrid=True, gridcolor='#EDEBE9')
    )

    ftr_rate_val     = filtered.ftr.mean() * 100
    loop_rate_val    = filtered.loop_flag.mean() * 100
    rework_cases_val = int(filtered.loop_flag.sum())
    multi_xfer_val   = (filtered.transfers >= 2).mean() * 100

    kpi_row = dbc.Row([
        dbc.Col([html.Div([html.H4("Direct Resolution Rate"),
                           html.H2(f"{ftr_rate_val:.1f}%")],
                          className="kpi-card kpi-success animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Loop / Rework Rate"),
                           html.H2(f"{loop_rate_val:.1f}%")],
                          className="kpi-card kpi-danger animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Cases with Rework"),
                           html.H2(f"{rework_cases_val:,}")],
                          className="kpi-card kpi-warning animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Multi-Transfer Cases (2+)"),
                           html.H2(f"{multi_xfer_val:.1f}%")],
                          className="kpi-card kpi-info animated-card")], md=3),
    ], className="mb-4")

    return html.Div([
        guide_statement([
            html.Strong("Not all queues add value, some just add delay. "),
            "The intermediary queues shown here are where Messenger cases sit waiting between handoffs, ",
            html.Strong("contributing nothing to resolution. "),
            "If a queue appears frequently in the Pareto, it's either a ",
            html.Strong("structural bottleneck"),
            " or a sign that cases are being sent there by mistake.",
        ]),
        kpi_row,
        html.Hr(className="divider"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=pareto_fig, config={'responsive': False})], md=6),
            dbc.Col([dcc.Graph(figure=entry_fig,  config={'responsive': False})], md=6),
        ]),
    ])


# ==================================
# TAB 3: COST & EFFORT IMPACT  (merged Cost Inflation + Customer Friction)
# ==================================

@callback(
    Output('impact-content', 'children'),
    [Input('impact-date-filter', 'start_date'), Input('impact-date-filter', 'end_date'),
     Input('impact-queue-filter', 'value'),    Input('impact-hours-filter', 'value'),
     Input('impact-segment-filter', 'value')]
)
def update_impact_tab(start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    # Cap at P95 to remove extreme outliers
    p95_aht = filtered['total_active_aht'].quantile(0.95)
    p95_msg = filtered['messages'].quantile(0.95)

    aht_0  = filtered[filtered.transfer_bin == '0' ]['total_active_aht'].median()
    aht_3p = filtered[filtered.transfer_bin == '3+']['total_active_aht'].median()
    msg_0  = filtered[filtered.transfer_bin == '0' ]['messages'].median()
    msg_3p = filtered[filtered.transfer_bin == '3+']['messages'].median()
    aht_pct = (aht_3p / aht_0 - 1) * 100 if aht_0 > 0 else 0
    msg_pct = (msg_3p / msg_0 - 1) * 100 if msg_0 > 0 else 0

    kpi_row = dbc.Row([
        dbc.Col([html.Div([html.H4("AHT — First Touch"), html.H2(f"{aht_0:.0f} min")],
                          className="kpi-card kpi-success animated-card")], md=3),
        dbc.Col([html.Div([html.H4("AHT — 3+ Transfers"), html.H2(f"{aht_3p:.0f} min")],
                          className="kpi-card kpi-danger animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Messages — First Touch"), html.H2(f"{msg_0:.0f}")],
                          className="kpi-card kpi-success animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Messages — 3+ Transfers"), html.H2(f"{msg_3p:.0f}")],
                          className="kpi-card kpi-danger animated-card")], md=3),
    ], className="mb-4")

    bin_colors = {
        '0': POWERBI_COLORS['success'],
        '1': POWERBI_COLORS['warning'],
        '2': '#E8820C',
        '3+': POWERBI_COLORS['danger']
    }

    # Box plots — AHT (capped at P95, no individual dots, mean + median labelled)
    aht_fig = go.Figure()
    for tbin in ['0', '1', '2', '3+']:
        data = filtered[filtered.transfer_bin == tbin]['total_active_aht']
        capped = data[data <= p95_aht].dropna()
        label = f"{tbin} transfer{'s' if tbin != '1' else ''}"
        if len(capped) > 0:
            aht_fig.add_trace(go.Box(
                y=capped,
                name=label,
                marker=dict(color=bin_colors[tbin]),
                line=dict(color=bin_colors[tbin], width=1.5),
                fillcolor=bin_colors[tbin],
                boxpoints=False,
                boxmean=True,
                whiskerwidth=0.6,
                opacity=0.35,
            ))
            med_val = capped.median()
            mean_val = capped.mean()
            aht_fig.add_annotation(x=label, y=med_val, text=f"Median: {med_val:.0f}",
                                   showarrow=False, xshift=70, font=dict(size=10, color=bin_colors[tbin]))
            aht_fig.add_annotation(x=label, y=mean_val, text=f"Mean: {mean_val:.0f}",
                                   showarrow=False, xshift=65, font=dict(size=10, color='#666'),
                                   yshift=12 if abs(mean_val - med_val) < p95_aht * 0.05 else 0)
    aht_fig.update_layout(
        title=dict(text="Handle Time Distribution by Transfer Count",
                   font=dict(size=13, color='#201F1E', family='Segoe UI')),
        yaxis_title="Active Handle Time (min)",
        width=540, height=460, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9', zeroline=False),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=50, r=80, t=60, b=40)
    )

    # Box plots — Messages (capped at P95, no individual dots, mean + median labelled)
    msg_fig = go.Figure()
    for tbin in ['0', '1', '2', '3+']:
        data = filtered[filtered.transfer_bin == tbin]['messages']
        capped = data[data <= p95_msg].dropna()
        label = f"{tbin} transfer{'s' if tbin != '1' else ''}"
        if len(capped) > 0:
            msg_fig.add_trace(go.Box(
                y=capped,
                name=label,
                marker=dict(color=bin_colors[tbin]),
                line=dict(color=bin_colors[tbin], width=1.5),
                fillcolor=bin_colors[tbin],
                boxpoints=False,
                boxmean=True,
                whiskerwidth=0.6,
                opacity=0.35,
            ))
            med_val = capped.median()
            mean_val = capped.mean()
            msg_fig.add_annotation(x=label, y=med_val, text=f"Median: {med_val:.0f}",
                                   showarrow=False, xshift=70, font=dict(size=10, color=bin_colors[tbin]))
            msg_fig.add_annotation(x=label, y=mean_val, text=f"Mean: {mean_val:.0f}",
                                   showarrow=False, xshift=65, font=dict(size=10, color='#666'),
                                   yshift=12 if abs(mean_val - med_val) < p95_msg * 0.05 else 0)
    msg_fig.update_layout(
        title=dict(text="Customer Messages by Transfer Count",
                   font=dict(size=13, color='#201F1E', family='Segoe UI')),
        yaxis_title="Messages from Customer",
        width=540, height=460, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9', zeroline=False),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=50, r=80, t=60, b=40)
    )

    # Dual escalation index (both AHT + Messages indexed to 100 at 0 transfers)
    esc = filtered.groupby('transfer_bin').agg(
        aht=('total_active_aht', 'median'),
        msg=('messages', 'median')
    ).reset_index()

    base_aht = esc.loc[esc.transfer_bin == '0', 'aht'].values
    base_msg = esc.loc[esc.transfer_bin == '0', 'msg'].values

    if len(base_aht) > 0 and base_aht[0] > 0:
        esc['aht_idx'] = esc['aht'] / base_aht[0] * 100
        esc['msg_idx'] = esc['msg'] / base_msg[0] * 100
    else:
        esc['aht_idx'] = esc['aht']
        esc['msg_idx'] = esc['msg']

    esc_fig = go.Figure()
    esc_fig.add_trace(go.Bar(
        x=esc['transfer_bin'], y=esc['aht_idx'], name='Handle Time (indexed)',
        marker=dict(color=POWERBI_COLORS['primary'], line=dict(width=0)),
        text=esc['aht_idx'].round(0), textposition='outside',
        texttemplate='%{text:.0f}', textfont=dict(size=13, color=POWERBI_COLORS['primary']),
    ))
    esc_fig.add_trace(go.Bar(
        x=esc['transfer_bin'], y=esc['msg_idx'], name='Customer Messages (indexed)',
        marker=dict(color=POWERBI_COLORS['warning'], line=dict(width=0)),
        text=esc['msg_idx'].round(0), textposition='outside',
        texttemplate='%{text:.0f}', textfont=dict(size=13, color=POWERBI_COLORS['warning']),
    ))
    esc_fig.add_hline(y=100, line_dash="dash", line_color="#999", line_width=1.5,
                      annotation_text="Baseline (0 transfers = 100)",
                      annotation_font=dict(size=11, color='#666'))
    esc_fig.update_layout(
        title=dict(text="Multiplier Effect: Handle Time & Messages vs First-Touch Baseline",
                   font=dict(size=13, color='#201F1E', family='Segoe UI')),
        xaxis_title="Number of Transfers",
        yaxis_title="Index (0 transfers = 100)",
        barmode='group',
        bargap=0.25, bargroupgap=0.08,
        width=1100, height=400, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9', zeroline=False),
        xaxis=dict(showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=11)),
        margin=dict(l=60, r=20, t=70, b=50)
    )

    insight = html.Div([
        html.P([
            "",
            html.Strong(f"Every additional transfer inflates handle time by ~{aht_pct/3:.0f}% per step "),
            f"and customer messages by ~{msg_pct/3:.0f}% per step. Cases reaching 3+ transfers carry ",
            html.Strong(f"{aht_pct:.0f}% more AHT"), f"and ",
            html.Strong(f"{msg_pct:.0f}% more customer messages"),
            "than first-touch resolutions."
        ], style={'margin': 0, 'fontSize': '0.92rem', 'color': '#333'})
    ], className="insight-card mb-3")

    return html.Div([
        guide_statement([
            "Every transfer doesn't just delay the customer, ",
            html.Strong("it inflates the total effort. "),
            "A case that gets transferred 3+ times costs ",
            html.Strong(f"{aht_pct:.0f}% more handle time"),
            " and generates ",
            html.Strong(f"{msg_pct:.0f}% more customer messages"),
            " than one resolved first-touch. ",
            html.Strong("This is the compounding cost of mis-routing."),
        ]),
        kpi_row,
        insight,
        html.Hr(className="divider"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=aht_fig, config={'responsive': False})], md=6),
            dbc.Col([dcc.Graph(figure=msg_fig, config={'responsive': False})], md=6),
        ], className="mb-2"),
        html.Hr(className="divider"),
        dcc.Graph(figure=esc_fig, config={'responsive': False})
    ])


# ==================================
# TAB 4: HOURS & TRANSFER EFFECT
# ==================================

@callback(
    Output('hours-content', 'children'),
    [Input('hours-date-filter', 'start_date'), Input('hours-date-filter', 'end_date'),
     Input('hours-queue-filter', 'value'),    Input('hours-hours-filter', 'value'),
     Input('hours-segment-filter', 'value')]
)
def update_hours_tab(start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    ih  = filtered[filtered.inhours == 1]
    ooh = filtered[filtered.inhours == 0]

    ih_multi  = (ih.transfers >= 2).mean() * 100 if len(ih) > 0 else 0
    ooh_multi = (ooh.transfers >= 2).mean() * 100 if len(ooh) > 0 else 0
    ih_aht    = ih.total_active_aht.median() if len(ih) > 0 else 0
    ooh_aht   = ooh.total_active_aht.median() if len(ooh) > 0 else 0
    ooh_aht_penalty = (ooh_aht / ih_aht - 1) * 100 if ih_aht > 0 else 0

    insight = html.Div([
        html.P([
            "Out-of-hours cases have ",
            html.Strong(f"{ooh_multi:.0f}% multi-transfer rate vs {ih_multi:.0f}% in-hours", style={'color': POWERBI_COLORS['danger']}),
            " — AND each transfer takes ",
            html.Strong(f"{ooh_aht_penalty:+.0f}% longer to handle.", style={'color': POWERBI_COLORS['danger']}),
            " The OOH penalty compounds with each successive transfer."
        ], style={'margin': 0, 'fontSize': '0.92rem', 'color': '#333'})
    ], className="insight-card mb-4")

    summary_cards = dbc.Row([
        dbc.Col([html.Div([html.H4("Multi-Transfer Rate (IH)"),  html.H2(f"{ih_multi:.0f}%")],
                          className="kpi-card kpi-success animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Multi-Transfer Rate (OOH)"), html.H2(f"{ooh_multi:.0f}%")],
                          className="kpi-card kpi-danger animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Median AHT (IH)"),           html.H2(f"{ih_aht:.0f} min")],
                          className="kpi-card kpi-success animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Median AHT (OOH)"),          html.H2(f"{ooh_aht:.0f} min")],
                          className="kpi-card kpi-danger animated-card")], md=3),
    ], className="mb-4")

    # Toggle buttons for heatmap views
    view_selector = html.Div([
        html.Div("Heatmap View", style={
            'fontSize': '0.7rem', 'fontWeight': '700', 'color': '#888',
            'textTransform': 'uppercase', 'letterSpacing': '0.8px',
            'marginBottom': '0.5rem',
        }),
        dbc.RadioItems(
            id='hours-heatmap-view',
            options=[
                {'label': 'Median AHT',         'value': 'aht'},
                {'label': 'Customer Messages',   'value': 'messages'},
                {'label': 'Transfer Volume',     'value': 'volume'},
                {'label': 'Median Routing Wait', 'value': 'routing'},
                {'label': 'In/Out Hours Split',  'value': 'inhours'},
            ],
            value='aht',
            inline=True,
            input_class_name="btn-check",
            label_class_name="btn btn-outline-primary btn-sm me-2",
            label_checked_class_name="active",
        ),
    ], style={'marginBottom': '1rem'})

    return html.Div([
        guide_statement([
            "Out-of-hours cases don't just transfer more often, ",
            html.Strong("they transfer harder. "),
            "The OOH multi-transfer rate is ",
            html.Strong(f"{ooh_multi:.0f}% vs {ih_multi:.0f}% in-hours"),
            ", and each of those transfers costs ",
            html.Strong(f"{ooh_aht_penalty:+.0f}% more handle time. "),
            "The heatmap below reveals exactly when the routing breaks down across the week.",
        ]),
        insight,
        summary_cards,
        html.Hr(className="divider"),
        view_selector,
        html.Div(id='hours-heatmap-output'),
    ])


DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
HOUR_LABELS = [f'{h:02d}:00' for h in range(24)]


@callback(
    Output('hours-heatmap-output', 'children'),
    [Input('hours-heatmap-view', 'value'),
     Input('hours-date-filter', 'start_date'), Input('hours-date-filter', 'end_date'),
     Input('hours-queue-filter', 'value'),    Input('hours-hours-filter', 'value'),
     Input('hours-segment-filter', 'value')]
)
def update_hours_heatmap(view, start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if len(filtered) == 0:
        return html.Div("No data for current filters.", className="alert alert-warning")

    # Build Day x Hour pivot — single red gradient for all views
    red_scale = [[0, '#FFF5F5'], [0.2, '#FFCDD2'], [0.4, '#EF9A9A'],
                 [0.6, '#E57373'], [0.8, '#D32F2F'], [1, '#8B0000']]
    hm_config = {
        'aht':      {'col': 'total_active_aht', 'agg': 'median', 'fmt': '.0f', 'unit': 'min',
                     'title': 'Median Handle Time (min) by Day and Hour',
                     'colorscale': red_scale},
        'messages': {'col': 'messages', 'agg': 'median', 'fmt': '.0f', 'unit': 'msgs',
                     'title': 'Median Customer Messages by Day and Hour',
                     'colorscale': red_scale},
        'volume':   {'col': 'CASE_ID', 'agg': 'count', 'fmt': '.0f', 'unit': 'cases',
                     'title': 'Transfer Volume (Case Count) by Day and Hour',
                     'colorscale': red_scale},
        'routing':  {'col': 'routing_days', 'agg': 'median', 'fmt': '.1f', 'unit': 'days',
                     'title': 'Median Routing Wait (days) by Day and Hour',
                     'colorscale': red_scale},
        'inhours':  {'col': 'inhours', 'agg': 'mean', 'fmt': '.0%', 'unit': '% IH',
                     'title': 'In-Hours Rate (%) by Day and Hour',
                     'colorscale': red_scale},
    }

    cfg = hm_config.get(view, hm_config['aht'])
    col, agg, fmt, unit = cfg['col'], cfg['agg'], cfg['fmt'], cfg['unit']

    pivot = (filtered.groupby(['day_of_week', 'hour_of_day'])[col]
             .agg(agg).reset_index())
    pivot_wide = pivot.pivot(index='day_of_week', columns='hour_of_day', values=col)
    # Ensure all days and hours present, zero-fill any gaps
    pivot_wide = pivot_wide.reindex(index=range(7), columns=range(24), fill_value=0)
    pivot_wide = pivot_wide.fillna(0)

    vals = pivot_wide.values
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    mid = (vmin + vmax) / 2 if vals.size > 0 else 0

    # Build annotations with smart text color
    annotations = []
    for i in range(7):
        for j in range(24):
            v = vals[i][j]
            if np.isnan(v):
                continue
            font_color = 'white' if v > mid else '#333'
            if view == 'inhours':
                text = f"{v:.0%}"
            elif fmt == '.1f':
                text = f"{v:.1f}"
            else:
                text = f"{v:.0f}"
            annotations.append(dict(
                x=HOUR_LABELS[j], y=DAY_NAMES[i], text=text,
                font=dict(size=10, family='Segoe UI', color=font_color),
                showarrow=False, xref='x', yref='y'
            ))

    fig = go.Figure(data=go.Heatmap(
        z=vals,
        x=HOUR_LABELS,
        y=DAY_NAMES,
        colorscale=cfg['colorscale'],
        showscale=True,
        colorbar=dict(title=dict(text=unit, font=dict(size=11)),
                      thickness=14, len=0.85, outlinewidth=0),
        xgap=2, ygap=2,
        hovertemplate='%{y}, %{x}<br>' + unit + ': %{z' + (':' + fmt if fmt != '.0%' else '') + '}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text=cfg['title'],
                   font=dict(size=14, color='#201F1E', family='Segoe UI')),
        xaxis=dict(title="Hour of Day", tickfont=dict(size=10), tickangle=0,
                   dtick=1, side='bottom'),
        yaxis=dict(tickfont=dict(size=11), autorange='reversed'),
        width=1200, height=500, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI'),
        margin=dict(l=100, r=60, t=55, b=50),
        annotations=annotations,
    )

    return html.Div([
        dcc.Graph(figure=fig, config={'responsive': False}),
        html.P(f"Showing {len(filtered):,} cases. Cells show {unit} values. Darker = higher.",
               style={'fontSize': '0.78rem', 'color': '#999', 'marginTop': '0.5rem',
                      'textAlign': 'center'}),
    ])


# ==================================
# TAB 5: QUEUE INTELLIGENCE  (merged Deep Dive + Transfer Flow)
# ==================================

@callback(
    Output('qi-content', 'children'),
    [Input('qi-date-filter', 'start_date'), Input('qi-date-filter', 'end_date'),
     Input('qi-queue-filter', 'value'),    Input('qi-hours-filter', 'value'),
     Input('qi-segment-filter', 'value')]
)
def update_qi_tab(start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    all_queues = sorted(df_raw.QUEUE_NEW.dropna().unique())
    return html.Div([
        guide_statement([
            "Every queue tells a story: ",
            html.Strong("is it resolving cases, or just passing them along? "),
            "Select a queue below to see who's sending it work, where it sends cases next, and how long they dwell. ",
            "If a queue has high inbound volume but low resolution, it's acting as ",
            html.Strong("an expensive middleman."),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([html.Span("SELECT QUEUE", style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': '#444', 'letterSpacing': '0.5px'
                    })], className="slicer-header"),
                    html.Div([
                        dcc.Dropdown(
                            id='qi-queue-selector',
                            options=[{'label': q, 'value': q} for q in all_queues],
                            value=all_queues[0] if all_queues else None,
                            clearable=False, style={'fontSize': '0.9rem'}
                        )
                    ], className="slicer-body")
                ], className="slicer-card")
            ], md=5)
        ], className="mb-4"),
        html.Div(id='qi-analysis')
    ])


@callback(
    Output('qi-analysis', 'children'),
    [Input('qi-queue-selector', 'value'),
     Input('qi-date-filter', 'start_date'),  Input('qi-date-filter', 'end_date'),
     Input('qi-queue-filter', 'value'),       Input('qi-hours-filter', 'value'),
     Input('qi-segment-filter', 'value')]
)
def update_qi_analysis(selected_queue, start_date, end_date, queues, hours, segments):
    if not selected_queue:
        return html.Div()

    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    filtered_cases = filtered.CASE_ID.unique()
    df_f = df_raw[df_raw.CASE_ID.isin(filtered_cases)]
    subset = df_f[df_f.QUEUE_NEW == selected_queue]

    if len(subset) == 0:
        return html.Div("No data for this queue in the current selection.", className="alert alert-warning")

    n_cases    = subset.CASE_ID.nunique()
    med_dwell  = subset.DAYS_IN_QUEUE.median()
    p90_dwell  = subset.DAYS_IN_QUEUE.quantile(0.9)
    total_delay= subset.DAYS_IN_QUEUE.sum()
    pct_delay  = total_delay / max(df_f.DAYS_IN_QUEUE.sum(), 1) * 100
    entry_cases = filtered[filtered.entry_queue == selected_queue]
    ftr_entry  = entry_cases.ftr.mean() * 100 if len(entry_cases) > 0 else 0
    final_cases = filtered[filtered.final_queue == selected_queue]
    pct_final  = len(final_cases) / max(n_cases, 1) * 100

    kpi_cards = dbc.Row([
        dbc.Col([html.Div([html.H4("Cases Through Queue"), html.H2(f"{n_cases:,}")],
                          className="kpi-card kpi-primary animated-card")], md=2),
        dbc.Col([html.Div([html.H4("Median Dwell (Days)"),  html.H2(f"{med_dwell:.1f}")],
                          className="kpi-card kpi-warning animated-card")], md=2),
        dbc.Col([html.Div([html.H4("P90 Dwell (Days)"),     html.H2(f"{p90_dwell:.1f}")],
                          className="kpi-card kpi-danger animated-card")], md=2),
        dbc.Col([html.Div([html.H4("FTR as Entry Queue"),   html.H2(f"{ftr_entry:.0f}%")],
                          className="kpi-card kpi-success animated-card")], md=3),
        dbc.Col([html.Div([html.H4("% of Total Routing Delay"), html.H2(f"{pct_delay:.1f}%")],
                          className="kpi-card kpi-info animated-card")], md=3),
    ], className="mb-4")

    # Dwell time histogram
    hist_fig = px.histogram(
        subset, x='DAYS_IN_QUEUE', nbins=25,
        title=f"Dwell Time Distribution: {selected_queue}",
        color_discrete_sequence=[POWERBI_COLORS['primary']]
    )
    hist_fig.add_vline(x=med_dwell, line_dash="dash", line_color=POWERBI_COLORS['danger'],
                       annotation_text=f"Median: {med_dwell:.1f}d", annotation_font_size=11)
    hist_fig.add_vline(x=p90_dwell, line_dash="dot", line_color=POWERBI_COLORS['warning'],
                       annotation_text=f"P90: {p90_dwell:.1f}d", annotation_font_size=11)
    hist_fig.update_layout(
        width=1100, height=320, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        xaxis_title="Days in Queue", yaxis_title="Number of Rows",
        margin=dict(l=50, r=20, t=60, b=40)
    )

    # Inbound & outbound flows
    inbound, outbound = [], []
    for cid in df_f.CASE_ID.unique():
        journey = df_f[df_f.CASE_ID == cid].sort_values('QUEUE_ORDER').QUEUE_NEW.tolist()
        for i, q in enumerate(journey):
            if q == selected_queue:
                if i > 0:              inbound.append(journey[i - 1])
                if i < len(journey)-1: outbound.append(journey[i + 1])

    def flow_chart(flows, title, color):
        if not flows:
            fig = go.Figure()
            fig.add_annotation(text="No transfers for this queue", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=13))
            fig.update_layout(title=title, width=540, height=320, autosize=False)
            return fig
        cnt = pd.Series(flows).value_counts().head(8).reset_index()
        cnt.columns = ['Queue', 'Cases']
        fig = go.Figure(data=[go.Bar(
            y=cnt['Queue'], x=cnt['Cases'], orientation='h',
            marker_color=color, text=cnt['Cases'], textposition='outside'
        )])
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, family='Segoe UI')),
            width=540, height=320, autosize=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Segoe UI'),
            xaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
            margin=dict(l=20, r=60, t=55, b=40)
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    in_fig  = flow_chart(inbound,  f"Top Sources Feeding {selected_queue}",       POWERBI_COLORS['secondary'])
    out_fig = flow_chart(outbound, f"Top Destinations After {selected_queue}",     POWERBI_COLORS['primary'])

    # Top transfer paths (full)
    all_paths = []
    for cid in df_f[df_f.QUEUE_NEW == selected_queue].CASE_ID.unique():
        j = df_f[df_f.CASE_ID == cid].sort_values('QUEUE_ORDER').QUEUE_NEW.tolist()
        all_paths.append('→ '.join(j))

    path_counts = pd.Series(all_paths).value_counts().head(10).reset_index()
    path_counts.columns = ['Journey Path', 'Cases']
    total_through = len(all_paths)
    path_counts['% of Cases'] = (path_counts['Cases'] / total_through * 100).round(1).astype(str) + '%'

    path_table = dbc.Table.from_dataframe(
        path_counts, striped=True, bordered=True, hover=True, responsive=True, className="mt-2"
    )

    return html.Div([
        kpi_cards,
        html.Hr(className="divider"),
        dcc.Graph(figure=hist_fig, config={'responsive': False}),
        html.Hr(className="divider"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=in_fig,  config={'responsive': False})], md=6),
            dbc.Col([dcc.Graph(figure=out_fig, config={'responsive': False})], md=6),
        ]),
        html.Hr(className="divider"),
        html.H6(f"Top 10 Complete Journey Paths Through {selected_queue}",
                style={'fontWeight': '600', 'color': '#201F1E'}),
        html.P(f"All {total_through:,} cases that touched this queue. Percentages sum to 100% across all paths.",
               className="text-muted", style={'fontSize': '0.85rem'}),
        path_table
    ])


# ==================================
# TAB 6: JOURNEY PATHWAYS
# ==================================

@callback(
    Output('journey-content', 'children'),
    [Input('journey-date-filter', 'start_date'), Input('journey-date-filter', 'end_date'),
     Input('journey-queue-filter', 'value'),    Input('journey-hours-filter', 'value'),
     Input('journey-segment-filter', 'value')]
)
def update_journey_tab(start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    all_queues = sorted(df_raw.QUEUE_NEW.dropna().unique())
    return html.Div([
        guide_statement([
            html.Strong("The shortest path to resolution is the cheapest one. "),
            "This tab maps how Messenger cases actually flow through the business: the most common routes, ",
            "the longest chains, and the unnecessary detours. ",
            html.Strong("Every extra hop on the journey is time, effort, and customer patience burned."),
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([html.Span("SELECT QUEUE TO ANALYSE", style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': '#444', 'letterSpacing': '0.5px'
                    })], className="slicer-header"),
                    html.Div([
                        dcc.Dropdown(
                            id='journey-queue-selector',
                            options=[{'label': q, 'value': q} for q in all_queues],
                            value=all_queues[0] if all_queues else None,
                            placeholder="Choose a queue...",
                            clearable=False, style={'fontSize': '0.9rem'}
                        )
                    ], className="slicer-body")
                ], className="slicer-card")
            ], md=5),
            dbc.Col([
                html.Div([
                    html.Div([html.Span("JOURNEY DEPTH (LEVELS)", style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': '#444', 'letterSpacing': '0.5px'
                    })], className="slicer-header"),
                    html.Div([
                        dcc.Slider(
                            id='journey-depth-slider',
                            min=2, max=5, value=3,
                            marks={i: str(i) for i in range(2, 6)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="slicer-body", style={'paddingTop': '12px'})
                ], className="slicer-card")
            ], md=5),
        ], className="mb-4"),

        html.Div(id='journey-analysis')
    ])


@callback(
    Output('journey-analysis', 'children'),
    [Input('journey-queue-selector', 'value'),
     Input('journey-depth-slider', 'value'),
     Input('journey-date-filter', 'start_date'), Input('journey-date-filter', 'end_date'),
     Input('journey-queue-filter', 'value'),    Input('journey-hours-filter', 'value'),
     Input('journey-segment-filter', 'value')]
)
def update_journey_analysis(selected_queue, depth, start_date, end_date, queues, hours, segments):
    if not selected_queue:
        return html.Div()

    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    filtered_cases = filtered.CASE_ID.unique()
    df_f = df_raw[df_raw.CASE_ID.isin(filtered_cases)]

    q_cases = df_f[df_f.QUEUE_NEW == selected_queue].CASE_ID.unique()
    q_journeys = df_f[df_f.CASE_ID.isin(q_cases)].sort_values(['CASE_ID', 'QUEUE_ORDER'])

    # Forward paths
    forward_paths = []
    for cid in q_cases:
        j = q_journeys[q_journeys.CASE_ID == cid].QUEUE_NEW.tolist()
        if selected_queue in j:
            idx = j.index(selected_queue)
            path = j[idx:idx + depth]
            if len(path) > 1:
                forward_paths.append(path)

    # Backward paths
    backward_paths = []
    for cid in q_cases:
        j = q_journeys[q_journeys.CASE_ID == cid].QUEUE_NEW.tolist()
        if selected_queue in j:
            end_idx = j.index(selected_queue)
            path = j[max(0, end_idx - depth + 1):end_idx + 1]
            if len(path) > 1:
                backward_paths.append(path)

    # Complete paths with case ID mapping
    path_to_cases = {}
    for cid in q_cases:
        j = df_f[df_f.CASE_ID == cid].sort_values('QUEUE_ORDER').QUEUE_NEW.tolist()
        path_str = ' → '.join(j)
        path_to_cases.setdefault(path_str, []).append(str(cid))

    complete_paths = []
    for cid in q_cases:
        j = df_f[df_f.CASE_ID == cid].sort_values('QUEUE_ORDER').QUEUE_NEW.tolist()
        complete_paths.append(' → '.join(j))

    total_through = len(complete_paths)
    path_series = pd.Series(complete_paths)
    top_paths = path_series.value_counts().head(10).reset_index()
    top_paths.columns = ['Journey Path', 'Cases']
    top_paths['% of Cases'] = (top_paths['Cases'] / total_through * 100).round(1)

    # Cost columns per path
    path_aht, path_routing, path_msgs = [], [], []
    for path_str in top_paths['Journey Path']:
        cids = path_to_cases.get(path_str, [])
        path_cases = filtered[filtered.CASE_ID.astype(str).isin(cids)]
        path_aht.append(path_cases['total_active_aht'].median() if len(path_cases) > 0 else 0)
        path_routing.append(path_cases['routing_days'].median() if len(path_cases) > 0 else 0)
        path_msgs.append(path_cases['messages'].median() if len(path_cases) > 0 else 0)
    top_paths['Med AHT (min)'] = [f"{v:.0f}" for v in path_aht]
    top_paths['Med Routing (days)'] = [f"{v:.1f}" for v in path_routing]
    top_paths['Med Messages'] = [f"{v:.0f}" for v in path_msgs]
    top_paths['% of Cases'] = top_paths['% of Cases'].apply(lambda v: f"{v:.1f}%")

    # Avoidable transfers: entry queue == final queue (round trip)
    avoidable_count = 0
    for path_str, cids in path_to_cases.items():
        queues_in_path = [q.strip() for q in path_str.split('→')]
        if len(queues_in_path) > 1 and queues_in_path[0] == queues_in_path[-1]:
            avoidable_count += len(cids)
    avoidable_pct = (avoidable_count / total_through * 100) if total_through > 0 else 0

    # Stats
    stats_cards = dbc.Row([
        dbc.Col([html.Div([html.H4("Cases Through Queue"),   html.H2(f"{len(q_cases):,}")],
                          className="kpi-card kpi-primary animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Unique Forward Paths"),  html.H2(f"{len(set(map(tuple, forward_paths)))}")],
                          className="kpi-card kpi-success animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Unique Backward Paths"), html.H2(f"{len(set(map(tuple, backward_paths)))}")],
                          className="kpi-card kpi-warning animated-card")], md=3),
        dbc.Col([html.Div([html.H4("Avg Journey Length"),
                           html.H2(f"{np.mean([p.count('→')+1 for p in complete_paths]):.1f}")],
                          className="kpi-card kpi-info animated-card")], md=3),
    ], className="mb-4")

    # Avoidable transfer callout
    avoidable_callout = html.Div()
    if avoidable_count > 0:
        avoidable_callout = html.Div([
            html.P([
                html.Strong(f"{avoidable_count} cases ({avoidable_pct:.1f}%) took a round trip", style={'color': '#D32F2F'}),
                f" back to the same queue they started in. These cases were transferred out and eventually "
                f"returned to the entry queue for resolution. Every one of these transfers was avoidable."
            ], style={'margin': 0, 'fontSize': '0.87rem', 'color': '#333'})
        ], className="insight-card mb-3")

    path_note = html.Div([
        html.P([
            html.Strong("Click any row "),
            "to see the individual case IDs for that pathway. Cost columns show the median AHT, routing wait, "
            f"and customer messages for cases on each path. Showing top 10 of {total_through:,} cases through ",
            html.Strong(selected_queue), ".",
        ], style={'margin': 0, 'fontSize': '0.87rem', 'color': '#333'})
    ], className="insight-card mb-3")

    # Build clickable table rows
    table_header = html.Thead(html.Tr([
        html.Th(c, style={'backgroundColor': '#0078D4', 'color': 'white', 'fontWeight': '700',
                          'fontSize': '0.75rem', 'textTransform': 'uppercase', 'letterSpacing': '0.4px',
                          'padding': '10px 12px', 'border': 'none'})
        for c in ['Journey Path', 'Cases', '% of Cases', 'Med AHT (min)', 'Med Routing (days)', 'Med Messages']
    ]))
    table_rows = []
    for i, row in top_paths.iterrows():
        table_rows.append(html.Tr([
            html.Td(row['Journey Path'], style={'fontSize': '0.82rem', 'maxWidth': '400px', 'wordWrap': 'break-word'}),
            html.Td(row['Cases'], style={'fontSize': '0.82rem', 'textAlign': 'center'}),
            html.Td(row['% of Cases'], style={'fontSize': '0.82rem', 'textAlign': 'center'}),
            html.Td(row['Med AHT (min)'], style={'fontSize': '0.82rem', 'textAlign': 'center'}),
            html.Td(row['Med Routing (days)'], style={'fontSize': '0.82rem', 'textAlign': 'center'}),
            html.Td(row['Med Messages'], style={'fontSize': '0.82rem', 'textAlign': 'center'}),
        ], id={'type': 'journey-path-row', 'index': i}, style={'cursor': 'pointer'},
           className="journey-clickable-row"))

    path_table = dbc.Table([table_header, html.Tbody(table_rows)],
                           striped=True, bordered=True, hover=True, responsive=True, className="mt-2")

    # Store path-to-cases mapping for the modal callback
    store_data = {row['Journey Path']: path_to_cases.get(row['Journey Path'], [])
                  for _, row in top_paths.iterrows()}

    forward_sankey  = create_sankey_from_paths(forward_paths,  f"Forward Journey from {selected_queue}")
    backward_sankey = create_sankey_from_paths(backward_paths, f"Backward Journey to {selected_queue}")

    return html.Div([
        dcc.Store(id='journey-path-store', data=store_data),
        stats_cards,
        avoidable_callout,
        html.Hr(className="divider"),

        html.H6(f"Forward View: Where do customers go FROM {selected_queue}?",
                style={'fontWeight': '600', 'color': '#201F1E'}),
        html.P("Paths customers take AFTER entering this queue.", className="text-muted"),
        dcc.Graph(figure=forward_sankey, config={'responsive': False}),

        html.Hr(className="divider"),

        html.H6(f"Backward View: How do customers arrive TO {selected_queue}?",
                style={'fontWeight': '600', 'color': '#201F1E'}),
        html.P("Paths customers took BEFORE reaching this queue.", className="text-muted"),
        dcc.Graph(figure=backward_sankey, config={'responsive': False}),

        html.Hr(className="divider"),

        html.H6(f"Top 10 Complete Journey Paths Through {selected_queue}",
                style={'fontWeight': '600', 'color': '#201F1E'}),
        path_note,
        path_table,

        # Modal for case detail popup
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id='journey-modal-title')),
            dbc.ModalBody(id='journey-modal-body'),
        ], id='journey-modal', size='xl', is_open=False),
    ])


def create_sankey_from_paths(paths, title):
    if not paths:
        fig = go.Figure()
        fig.add_annotation(text="No journey data available for this selection",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14))
        fig.update_layout(title=title, width=1100, height=400, autosize=False)
        return fig

    links = []
    for path in paths:
        for i in range(len(path) - 1):
            links.append((f"{path[i]} (Step {i+1})", f"{path[i+1]} (Step {i+2})"))

    link_counts = pd.Series(links).value_counts().reset_index()
    link_counts.columns = ['link', 'count']
    link_counts[['source', 'target']] = pd.DataFrame(link_counts['link'].tolist(), index=link_counts.index)

    all_nodes = list(set(link_counts['source'].tolist() + link_counts['target'].tolist()))
    node_dict = {n: i for i, n in enumerate(all_nodes)}

    colors = px.colors.qualitative.Set3
    node_colors = [colors[i % len(colors)] for i in range(len(all_nodes))]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes, color=node_colors
        ),
        link=dict(
            source=[node_dict[s] for s in link_counts['source']],
            target=[node_dict[t] for t in link_counts['target']],
            value=link_counts['count'].tolist(),
            label=[f"{v} cases" for v in link_counts['count']]
        )
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color='#2c3e50', family='Segoe UI')),
        font=dict(size=11, family='Segoe UI'),
        width=1100, height=580,
        margin=dict(l=20, r=20, t=60, b=20),
        autosize=False
    )
    return fig


@callback(
    [Output('journey-modal', 'is_open'),
     Output('journey-modal-title', 'children'),
     Output('journey-modal-body', 'children')],
    Input({'type': 'journey-path-row', 'index': ALL}, 'n_clicks'),
    State('journey-path-store', 'data'),
    prevent_initial_call=True
)
def open_journey_modal(n_clicks, store_data):
    if not any(n_clicks) or not store_data:
        return False, "", ""

    # Find which row was clicked
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return False, "", ""
    row_idx = triggered['index']

    # Get the path string and case IDs
    paths = list(store_data.keys())
    if row_idx >= len(paths):
        return False, "", ""
    path_str = paths[row_idx]
    case_ids = store_data[path_str]

    # Build case detail table from case_df
    detail = case_df[case_df.CASE_ID.astype(str).isin(case_ids)].copy()
    if len(detail) == 0:
        return True, f"Cases on: {path_str}", html.P("No case data found.")

    show_cols = ['CASE_ID', 'entry_queue', 'final_queue', 'transfers', 'total_active_aht',
                 'routing_days', 'messages', 'segment']
    # Add ML columns if available
    for ml_col in ['transfer_risk', 'recommended_queue', 'cluster_name']:
        if ml_col in detail.columns:
            show_cols.append(ml_col)
    detail_show = detail[[c for c in show_cols if c in detail.columns]].copy()

    # Friendly column names
    rename = {'CASE_ID': 'Case ID', 'entry_queue': 'Entry Queue', 'final_queue': 'Final Queue',
              'transfers': 'Transfers', 'total_active_aht': 'AHT (min)', 'routing_days': 'Routing Days',
              'messages': 'Messages', 'segment': 'Segment', 'transfer_risk': 'Transfer Risk %',
              'recommended_queue': 'Recommended Queue', 'cluster_name': 'Cluster'}
    detail_show = detail_show.rename(columns=rename)
    if 'AHT (min)' in detail_show.columns:
        detail_show['AHT (min)'] = detail_show['AHT (min)'].round(0)
    if 'Routing Days' in detail_show.columns:
        detail_show['Routing Days'] = detail_show['Routing Days'].round(1)

    # Summary stats
    n = len(detail)
    med_aht = detail['total_active_aht'].median() if 'total_active_aht' in detail.columns else 0
    med_routing = detail['routing_days'].median() if 'routing_days' in detail.columns else 0
    med_msgs = detail['messages'].median() if 'messages' in detail.columns else 0

    summary = html.Div([
        dbc.Row([
            dbc.Col([html.Div([html.H4("Cases"), html.H2(f"{n}")],
                              className="kpi-card kpi-primary")], md=3),
            dbc.Col([html.Div([html.H4("Med AHT"), html.H2(f"{med_aht:.0f} min")],
                              className="kpi-card kpi-danger")], md=3),
            dbc.Col([html.Div([html.H4("Med Routing"), html.H2(f"{med_routing:.1f} days")],
                              className="kpi-card kpi-warning")], md=3),
            dbc.Col([html.Div([html.H4("Med Messages"), html.H2(f"{med_msgs:.0f}")],
                              className="kpi-card kpi-info")], md=3),
        ], className="mb-3"),
    ])

    table = dbc.Table.from_dataframe(
        detail_show, striped=True, bordered=True, hover=True, responsive=True,
        style={'fontSize': '0.8rem'}
    )

    hops = len([q.strip() for q in path_str.split('→')])
    title = f"{n} Cases on {hops}-Queue Path"

    return True, title, html.Div([summary, table])


# ==================================
# TAB 7: DATA EXPLORER
# ==================================

# Shared DataTable styling — Power BI-inspired
DT_STYLE_HEADER = {
    'backgroundColor': '#0078D4',
    'color': 'white',
    'fontWeight': '700',
    'fontSize': '0.75rem',
    'textTransform': 'uppercase',
    'letterSpacing': '0.4px',
    'border': 'none',
    'padding': '10px 12px',
    'fontFamily': 'Segoe UI, sans-serif',
}
DT_STYLE_DATA = {
    'backgroundColor': 'white',
    'color': '#201F1E',
    'fontSize': '0.83rem',
    'fontFamily': 'Segoe UI, sans-serif',
    'border': '1px solid #EDEBE9',
    'padding': '8px 12px',
}
DT_STYLE_CONDITIONAL = [
    {'if': {'row_index': 'odd'}, 'backgroundColor': '#F8F8F8'},
    {'if': {'state': 'selected'},
     'backgroundColor': 'rgba(0,188,242,0.08)', 'border': '1px solid #00BCF2'},
]


def build_view_df(view, filtered, df_raw_filtered):
    """Return (dataframe, filename) for a given view type."""
    if view == 'case':
        all_cols = [
            'CASE_ID', 'entry_queue', 'final_queue', 'transfers', 'transfer_bin',
            'total_active_aht', 'routing_days', 'close_hours', 'messages',
            'ftr', 'inhours', 'loop_flag',
            'segment',
            'transfer_risk', 'recommended_queue', 'queue_match', 'cluster_name',
        ]
        cols = [c for c in all_cols if c in filtered.columns]
        df = filtered[cols].copy()
        rename_map = {
            'CASE_ID': 'Case ID', 'entry_queue': 'Entry Queue',
            'final_queue': 'Final Queue', 'transfers': 'Transfers',
            'transfer_bin': 'Transfer Group', 'total_active_aht': 'AHT (min)',
            'routing_days': 'Routing Days', 'close_hours': 'Close Hours',
            'messages': 'Cust. Messages', 'ftr': 'Direct Resolved',
            'inhours': 'In-Hours', 'loop_flag': 'Has Loop',
            'segment': 'Segment',
            'transfer_risk': 'Transfer Risk %', 'recommended_queue': 'Recommended Queue',
            'queue_match': 'Queue Match', 'cluster_name': 'Journey Cluster',
        }
        df.columns = [rename_map.get(c, c) for c in df.columns]
        df['Direct Resolved'] = df['Direct Resolved'].map({1: 'Yes', 0: 'No'})
        df['In-Hours']        = df['In-Hours'].map({1: 'Yes', 0: 'No'})
        df['Has Loop']        = df['Has Loop'].map({1: 'Yes', 0: 'No'})
        df['AHT (min)']       = df['AHT (min)'].round(1)
        df['Routing Days']    = df['Routing Days'].round(2)
        df['Close Hours']     = df['Close Hours'].round(1)
        if 'Queue Match' in df.columns:
            df['Queue Match'] = df['Queue Match'].map({1: 'Yes', 0: 'No'})
        return df, 'messenger_case_summary.csv'

    elif view == 'journey':
        df = df_raw_filtered[[
            'CASE_ID', 'QUEUE_ORDER', 'QUEUE_NEW', 'DAYS_IN_QUEUE', 'TOTALACTIVEAHT', 'INHOURS'
        ]].copy()
        df.columns = ['Case ID', 'Queue Step', 'Queue Name', 'Days in Queue', 'Cumul. AHT (min)', 'In-Hours']
        df['Days in Queue']     = df['Days in Queue'].round(2)
        df['Cumul. AHT (min)'] = df['Cumul. AHT (min)'].round(1)
        df['In-Hours']          = df['In-Hours'].map({1: 'Yes', 0: 'No'})
        return df.sort_values(['Case ID', 'Queue Step']), 'messenger_queue_journey.csv'

    elif view == 'transfer':
        grp = filtered.groupby('transfer_bin', observed=True).agg(
            Cases=('CASE_ID', 'count'),
            Median_AHT=('total_active_aht', 'median'),
            Median_Messages=('messages', 'median'),
            Median_Routing_Days=('routing_days', 'median'),
            DRR=('ftr', 'mean'),
        ).reset_index()
        grp['% of Total']        = (grp['Cases'] / grp['Cases'].sum() * 100).round(1).astype(str) + '%'
        grp['Median AHT (min)']  = grp['Median_AHT'].round(1)
        grp['Median Messages']   = grp['Median_Messages'].round(1)
        grp['Median Routing Days'] = grp['Median_Routing_Days'].round(2)
        grp['Direct Resolution %'] = (grp['DRR'] * 100).round(1).astype(str) + '%'
        df = grp[['transfer_bin', 'Cases', '% of Total', 'Median AHT (min)',
                  'Median Messages', 'Median Routing Days', 'Direct Resolution %']].copy()
        df.columns = ['Transfer Count', 'Cases', '% of Total', 'Median AHT (min)',
                      'Median Messages', 'Median Routing Days', 'Direct Resolution %']
        return df, 'messenger_transfer_breakdown.csv'

    else:  # queue performance
        grp = filtered.groupby('entry_queue').agg(
            Cases=('CASE_ID', 'count'),
            DRR=('ftr', 'mean'),
            Avg_Transfers=('transfers', 'mean'),
            Median_AHT=('total_active_aht', 'median'),
            Median_Routing=('routing_days', 'median'),
            Loop_Rate=('loop_flag', 'mean'),
        ).reset_index()
        grp['Direct Resolution %'] = (grp['DRR'] * 100).round(1).astype(str) + '%'
        grp['Avg Transfers']        = grp['Avg_Transfers'].round(2)
        grp['Median AHT (min)']     = grp['Median_AHT'].round(1)
        grp['Median Routing Days']  = grp['Median_Routing'].round(2)
        grp['Loop Rate %']          = (grp['Loop_Rate'] * 100).round(1).astype(str) + '%'
        df = grp[['entry_queue', 'Cases', 'Direct Resolution %', 'Avg Transfers',
                  'Median AHT (min)', 'Median Routing Days', 'Loop Rate %']].copy()
        df.columns = ['Entry Queue', 'Cases', 'Direct Resolution %', 'Avg Transfers',
                      'Median AHT (min)', 'Median Routing Days', 'Loop Rate %']
        return df.sort_values('Cases', ascending=False), 'messenger_queue_performance.csv'


@callback(
    Output('explorer-content', 'children'),
    [Input('explorer-date-filter', 'start_date'), Input('explorer-date-filter', 'end_date'),
     Input('explorer-queue-filter', 'value'),    Input('explorer-hours-filter', 'value'),
     Input('explorer-segment-filter', 'value')]
)
def update_explorer_tab(start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if len(filtered) == 0:
        return html.Div("No data for selected filters.", className="alert alert-warning")

    n_cases = len(filtered)

    # ── Extra Transfer Count slicer ──────────────────────────────────────────
    xfer_slicer = html.Div([
        html.Div([
            html.Span("TRANSFER COUNT", style={
                'fontSize': '0.7rem', 'fontWeight': '700', 'color': '#444', 'letterSpacing': '0.5px'
            })
        ], className="slicer-header"),
        html.Div([
            dcc.Dropdown(
                id='explorer-xfer-filter',
                options=[
                    {'label': '0 — Direct Resolution', 'value': '0'},
                    {'label': '1 Transfer',             'value': '1'},
                    {'label': '2 Transfers',            'value': '2'},
                    {'label': '3+ Transfers',           'value': '3+'},
                ],
                value=['0', '1', '2', '3+'],
                multi=True,
                placeholder="All transfer counts...",
                style={'fontSize': '0.82rem'}
            )
        ], className="slicer-body")
    ], className="slicer-card mb-4", style={'maxWidth': '360px'})

    # ── View selector — styled as button tabs ─────────────────────────────────
    view_selector = html.Div([
        html.Div("Select View", style={
            'fontSize': '0.7rem', 'fontWeight': '700', 'color': '#888',
            'textTransform': 'uppercase', 'letterSpacing': '0.8px',
            'marginBottom': '0.5rem',
        }),
        dbc.RadioItems(
            id='explorer-view',
            options=[
                {'label': 'Case Summary',         'value': 'case'},
                {'label': 'Queue Journey (raw)',   'value': 'journey'},
                {'label': 'Transfer Breakdown',    'value': 'transfer'},
                {'label': 'Queue Performance',     'value': 'queue'},
            ],
            value='case',
            inline=True,
            input_class_name="btn-check",
            label_class_name="btn btn-outline-primary btn-sm me-2",
            label_checked_class_name="active",
        ),
    ], style={'marginBottom': '1rem'})

    # ── Download button ───────────────────────────────────────────────────────
    download_bar = html.Div([
        dbc.Button(
            "Download Current View as CSV",
            id='btn-explorer-download',
            color='primary', outline=True, size='sm',
            style={'fontSize': '0.8rem', 'fontWeight': '600'},
            n_clicks=0,
        ),
        html.Span(f"{n_cases:,} cases in current filter",
                  style={'fontSize': '0.78rem', 'color': '#888', 'marginLeft': '1rem',
                         'verticalAlign': 'middle'}),
        dcc.Download(id='explorer-download'),
    ], style={'marginBottom': '1rem', 'display': 'flex', 'alignItems': 'center'})

    return html.Div([
        guide_statement([
            html.Strong("Everything in this report is built from the data below. "),
            "Browse case-level summaries, queue-level detail, or full transfer paths, then download the CSV ",
            "to run your own analysis. ",
            html.Strong("No black boxes."),
        ]),
        xfer_slicer,
        view_selector,
        download_bar,
        html.Div(id='explorer-table-view'),
    ])


@callback(
    Output('explorer-table-view', 'children'),
    [Input('explorer-view', 'value'),
     Input('explorer-xfer-filter', 'value'),
     Input('explorer-date-filter', 'start_date'), Input('explorer-date-filter', 'end_date'),
     Input('explorer-queue-filter', 'value'),    Input('explorer-hours-filter', 'value'),
     Input('explorer-segment-filter', 'value')]
)
def update_explorer_table(view, xfer_bins, start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if len(filtered) == 0:
        return html.Div("No data.", className="alert alert-warning")

    if xfer_bins:
        filtered = filtered[filtered.transfer_bin.isin(xfer_bins)]
    if len(filtered) == 0:
        return html.Div("No data for selected transfer counts.", className="alert alert-warning")

    filtered_cases = filtered.CASE_ID.unique()
    df_raw_f = df_raw[df_raw.CASE_ID.isin(filtered_cases)]

    df, _ = build_view_df(view, filtered, df_raw_f)

    view_labels = {
        'case':     ('Case Summary', f'{len(df):,} rows — one row per Messenger case'),
        'journey':  ('Queue Journey (Raw)', f'{len(df):,} rows — one row per queue stop'),
        'transfer': ('Transfer Breakdown', 'Aggregated by transfer count group'),
        'queue':    ('Queue Performance', 'Aggregated by entry queue'),
    }
    label, sub = view_labels.get(view, ('View', ''))

    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in df.columns],
        sort_action='native',
        filter_action='native',
        page_action='native',
        page_size=25,
        style_table={'overflowX': 'auto', 'borderRadius': '6px',
                     'boxShadow': '0 1.6px 3.6px 0 rgba(0,0,0,.132)'},
        style_header=DT_STYLE_HEADER,
        style_data=DT_STYLE_DATA,
        style_data_conditional=DT_STYLE_CONDITIONAL,
        style_cell={'textAlign': 'left', 'minWidth': '90px', 'maxWidth': '260px',
                    'overflow': 'hidden', 'textOverflow': 'ellipsis'},
        style_filter={
            'backgroundColor': '#EEF8FF', 'fontSize': '0.78rem',
            'fontFamily': 'Segoe UI, sans-serif', 'border': '1px solid #C8DFEF',
        },
        tooltip_delay=0,
        tooltip_duration=None,
        filter_options={'case': 'insensitive'},
    )

    return html.Div([
        html.Div([
            html.Span(label, style={'fontWeight': '700', 'fontSize': '0.9rem', 'color': '#201F1E'}),
            html.Span(f'— {sub}',
                      style={'fontSize': '0.78rem', 'color': '#888', 'marginLeft': '0.3rem'}),
        ], style={'marginBottom': '0.6rem'}),
        html.P("Use column headers to sort. Type in the filter row (grey) to search within any column.",
               style={'fontSize': '0.75rem', 'color': '#AAA', 'marginBottom': '0.7rem'}),
        table,
    ])


@callback(
    Output('explorer-download', 'data'),
    Input('btn-explorer-download', 'n_clicks'),
    [State('explorer-view', 'value'),
     State('explorer-xfer-filter', 'value'),
     State('explorer-date-filter', 'start_date'), State('explorer-date-filter', 'end_date'),
     State('explorer-queue-filter', 'value'),     State('explorer-hours-filter', 'value'),
     State('explorer-segment-filter', 'value')],
    prevent_initial_call=True,
)
def download_explorer_data(n_clicks, view, xfer_bins, start_date, end_date, queues, hours, segments):
    filtered = filter_data(case_df, start_date, end_date, queues, hours, segments)
    if xfer_bins:
        filtered = filtered[filtered.transfer_bin.isin(xfer_bins)]
    filtered_cases = filtered.CASE_ID.unique()
    df_raw_f = df_raw[df_raw.CASE_ID.isin(filtered_cases)]
    df, filename = build_view_df(view or 'case', filtered, df_raw_f)
    return dcc.send_data_frame(df.to_csv, filename, index=False)


# ==================================
# TAB 8: ML INSIGHTS
# ==================================

def build_ml_insights_tab():
    """Static Tab 8 — model comparison, feature importances, cluster profiles."""
    art1 = ml_artifacts['model1']
    art2 = ml_artifacts['model2']
    art3 = ml_artifacts['model3']

    # MODEL 1: Transfer Risk
    m1_names = list(art1['all_scores'].keys())
    m1_aucs  = [art1['all_scores'][n]['mean'] for n in m1_names]
    m1_stds  = [art1['all_scores'][n]['std'] for n in m1_names]
    m1_colors = [POWERBI_COLORS['primary'] if n == art1['best_name'] else '#C8C6C4' for n in m1_names]

    fig_m1_compare = go.Figure(go.Bar(
        x=m1_names, y=m1_aucs, error_y=dict(type='data', array=m1_stds, visible=True),
        marker_color=m1_colors, text=[f"{v:.3f}" for v in m1_aucs], textposition='outside',
    ))
    fig_m1_compare.update_layout(
        title="Model Comparison — 5-Fold CV AUC", yaxis_title="AUC Score",
        width=360, height=380, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9', range=[0, 1.05]),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    imp_df = pd.DataFrame({
        'Feature': art1['feature_names'], 'Importance': art1['importances']
    }).sort_values('Importance', ascending=True).tail(10)
    imp_df['Feature'] = imp_df['Feature'].str.replace('eq_', '', regex=False)

    fig_m1_imp = go.Figure(go.Bar(
        y=imp_df['Feature'], x=imp_df['Importance'],
        orientation='h', marker_color=POWERBI_COLORS['primary'],
    ))
    fig_m1_imp.update_layout(
        title="Top 10 Feature Importances", width=380, height=380, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        xaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        margin=dict(l=130, r=20, t=50, b=40),
    )

    fig_m1_dist = go.Figure(go.Histogram(
        x=case_df['transfer_risk'], nbinsx=20,
        marker_color=POWERBI_COLORS['danger'], opacity=0.8,
    ))
    fig_m1_dist.update_layout(
        title="Transfer Risk Score Distribution",
        xaxis_title="Transfer Risk %", yaxis_title="Cases",
        width=360, height=380, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    model1_section = html.Div([
        html.Div([
            html.H6("Model 1: Transfer Risk Prediction", style={
                'fontWeight': '700', 'color': POWERBI_COLORS['primary'], 'marginBottom': '0.3rem'}),
            html.P([
                f"Best model: ", html.Strong(art1['best_name']),
                f" — 5-fold CV AUC: {art1['cv_auc_mean']:.3f} (+/-{art1['cv_auc_std']:.3f})",
            ], style={'fontSize': '0.85rem', 'color': '#555', 'marginBottom': '0'}),
            html.P("Predicts the probability a Messenger case will require at least one transfer, "
                   "using only information available at case creation (entry queue, in/out hours, day, time).",
                   style={'fontSize': '0.8rem', 'color': '#888', 'marginBottom': '0'}),
        ], className="insight-card mb-3"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_m1_compare, config={'responsive': False})], md=4),
            dbc.Col([dcc.Graph(figure=fig_m1_imp, config={'responsive': False})], md=4),
            dbc.Col([dcc.Graph(figure=fig_m1_dist, config={'responsive': False})], md=4),
        ]),
    ])

    # MODEL 2: Queue Recommendation
    m2_names = list(art2['all_scores'].keys())
    m2_accs  = [art2['all_scores'][n]['mean'] for n in m2_names]
    m2_stds  = [art2['all_scores'][n]['std'] for n in m2_names]
    m2_colors = [POWERBI_COLORS['success'] if n == art2['best_name'] else '#C8C6C4' for n in m2_names]

    fig_m2_compare = go.Figure(go.Bar(
        x=m2_names, y=[a * 100 for a in m2_accs],
        error_y=dict(type='data', array=[s * 100 for s in m2_stds], visible=True),
        marker_color=m2_colors,
        text=[f"{v*100:.1f}%" for v in m2_accs], textposition='outside',
    ))
    fig_m2_compare.update_layout(
        title="Model Comparison — 5-Fold CV Accuracy", yaxis_title="Accuracy %",
        width=540, height=380, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9', range=[0, 105]),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    match_rate = case_df['queue_match'].mean() * 100
    reroutable = int(case_df['queue_match'].eq(0).sum())
    transferred = case_df[case_df.transfers > 0]
    reroute_match = transferred['queue_match'].mean() * 100 if len(transferred) > 0 else 0

    m2_kpis = dbc.Row([
        dbc.Col([html.Div([html.H4("Recommendation Match Rate"),
                           html.H2(f"{match_rate:.1f}%")],
                          className="kpi-card kpi-success animated-card")], md=4),
        dbc.Col([html.Div([html.H4("Cases Could Be Re-routed"),
                           html.H2(f"{reroutable:,}")],
                          className="kpi-card kpi-warning animated-card")], md=4),
        dbc.Col([html.Div([html.H4("Match on Transferred Cases"),
                           html.H2(f"{reroute_match:.1f}%")],
                          className="kpi-card kpi-info animated-card")], md=4),
    ], className="mb-3")

    misroutes = case_df[
        (case_df['entry_queue'] != case_df['final_queue']) & (case_df['queue_match'] == 0)
    ]
    if len(misroutes) > 0:
        misroute_top = (misroutes.groupby(['entry_queue', 'final_queue'])
                        .size().reset_index(name='count')
                        .sort_values('count', ascending=False).head(8))
        fig_misroute = go.Figure(go.Bar(
            x=misroute_top.apply(lambda r: f"{r['entry_queue']} -> {r['final_queue']}", axis=1),
            y=misroute_top['count'], marker_color=POWERBI_COLORS['warning'],
            text=misroute_top['count'], textposition='outside',
        ))
    else:
        fig_misroute = go.Figure()
        fig_misroute.add_annotation(text="No misroutes detected", xref="paper", yref="paper",
                                    x=0.5, y=0.5, showarrow=False)
    fig_misroute.update_layout(
        title="Top Misrouted Paths (Entry -> Actual Resolution Queue)",
        width=540, height=380, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        xaxis=dict(tickangle=-30), margin=dict(l=50, r=20, t=50, b=100),
    )

    confusion = pd.crosstab(case_df['final_queue'], case_df['recommended_queue'])
    fig_confusion = go.Figure(go.Heatmap(
        z=confusion.values, x=confusion.columns.tolist(), y=confusion.index.tolist(),
        colorscale=[[0, '#F3F2F1'], [1, POWERBI_COLORS['primary']]],
        text=confusion.values, texttemplate='%{text}', textfont=dict(size=10),
    ))
    fig_confusion.update_layout(
        title="Recommended vs Actual Resolution Queue",
        xaxis_title="Recommended Queue", yaxis_title="Actual Final Queue",
        width=540, height=480, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E', size=10),
        margin=dict(l=130, r=20, t=50, b=120), xaxis=dict(tickangle=-45),
    )

    model2_section = html.Div([
        html.Div([
            html.H6("Model 2: Optimal First-Queue Recommendation", style={
                'fontWeight': '700', 'color': POWERBI_COLORS['success'], 'marginBottom': '0.3rem'}),
            html.P([
                f"Best model: ", html.Strong(art2['best_name']),
                f" — 5-fold CV Accuracy: {art2['cv_acc_mean']*100:.1f}% (+/-{art2['cv_acc_std']*100:.1f}%)",
            ], style={'fontSize': '0.85rem', 'color': '#555', 'marginBottom': '0'}),
            html.P("Predicts which queue will ultimately resolve the case. If the model is right, "
                   "routing directly to that queue eliminates every intermediate transfer.",
                   style={'fontSize': '0.8rem', 'color': '#888', 'marginBottom': '0'}),
        ], className="insight-card mb-3"),
        dbc.Row([dbc.Col([dcc.Graph(figure=fig_m2_compare, config={'responsive': False})], md=6),
                 dbc.Col([m2_kpis], md=6)]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_misroute, config={'responsive': False})], md=6),
            dbc.Col([dcc.Graph(figure=fig_confusion, config={'responsive': False})], md=6),
        ]),
    ])

    # MODEL 3: Journey Clustering
    pca_df = pd.DataFrame({
        'PC1': art3['pca_coords'][:, 0], 'PC2': art3['pca_coords'][:, 1],
        'Cluster': case_df['cluster_name'], 'Case': case_df['CASE_ID'].astype(str),
        'Transfers': case_df['transfers'], 'AHT': case_df['total_active_aht'].round(0),
    })
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                         hover_data=['Case', 'Transfers', 'AHT'],
                         color_discrete_sequence=CHART_COLORS)
    fig_pca.update_layout(
        title=f"Clustering & Anomaly Detection (PCA 2D) — Silhouette: {art3['silhouette']:.2f}",
        width=600, height=480, autosize=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        xaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        margin=dict(l=50, r=20, t=60, b=40),
    )

    profiles = art3['cluster_profiles']
    card_colors = CHART_COLORS[:len(profiles)]
    profile_cards = []
    for i, (idx, row) in enumerate(profiles.iterrows()):
        cname = art3['name_map'][idx]
        profile_cards.append(
            dbc.Col([html.Div([
                html.Div(cname, style={'fontWeight': '700', 'color': card_colors[i],
                                       'fontSize': '0.9rem', 'marginBottom': '0.3rem'}),
                html.Div(f"{int(row['count'])} cases",
                         style={'fontSize': '0.75rem', 'color': '#888', 'marginBottom': '0.5rem'}),
                html.Div([
                    html.Div(f"Avg Transfers: {row['avg_transfers']:.1f}", style={'fontSize': '0.8rem', 'marginBottom': '0.15rem'}),
                    html.Div(f"Avg Routing Days: {row['avg_routing']:.1f}", style={'fontSize': '0.8rem', 'marginBottom': '0.15rem'}),
                    html.Div(f"Avg AHT: {row['avg_aht']:.0f} min", style={'fontSize': '0.8rem', 'marginBottom': '0.15rem'}),
                    html.Div(f"Avg Messages: {row['avg_messages']:.1f}", style={'fontSize': '0.8rem', 'marginBottom': '0.15rem'}),
                    html.Div(f"Loop Rate: {row['loop_rate']*100:.0f}%", style={'fontSize': '0.8rem'}),
                ], style={'color': '#605E5C'}),
            ], style={
                'background': 'white', 'borderRadius': '8px', 'padding': '1rem 1.2rem',
                'height': '100%', 'boxShadow': '0 1.6px 3.6px 0 rgba(0,0,0,.132)',
                'borderTop': f'3px solid {card_colors[i]}',
            })], md=12 // max(len(profiles), 1), className="mb-3")
        )

    # Identify anomaly clusters (avg_transfers >= 3)
    anomaly_clusters = [art3['name_map'][idx] for idx, row in profiles.iterrows()
                        if row['avg_transfers'] >= 3]
    anomaly_count = sum(int(row['count']) for idx, row in profiles.iterrows()
                        if row['avg_transfers'] >= 3)
    anomaly_pct = (anomaly_count / len(case_df) * 100) if len(case_df) > 0 else 0

    # Build dynamic cluster narrative
    total_cases = len(case_df)
    sorted_by_size = profiles.sort_values('count', ascending=False)
    narrative_items = []
    for idx, row in sorted_by_size.iterrows():
        cname = art3['name_map'][idx]
        count = int(row['count'])
        pct = count / total_cases * 100 if total_cases > 0 else 0
        is_anomaly = row['avg_transfers'] >= 3
        badge = html.Span(" ANOMALY", style={
            'fontSize': '0.65rem', 'fontWeight': '700', 'color': 'white',
            'background': '#D32F2F', 'borderRadius': '3px', 'padding': '1px 5px',
            'marginLeft': '0.4rem', 'verticalAlign': 'middle',
        }) if is_anomaly else None

        desc_parts = []
        if row['avg_transfers'] < 0.5:
            desc_parts.append(f"These {count} cases ({pct:.0f}%) were resolved with almost no transfers. ")
            desc_parts.append("This is the ideal routing outcome.")
        elif row['avg_transfers'] < 2:
            desc_parts.append(f"{count} cases ({pct:.0f}%) with moderate transfer activity. ")
            desc_parts.append(f"Averaging {row['avg_transfers']:.1f} transfers and {row['avg_aht']:.0f} min AHT.")
        else:
            desc_parts.append(f"{count} cases ({pct:.0f}%) averaging {row['avg_transfers']:.1f} transfers, ")
            desc_parts.append(f"{row['avg_aht']:.0f} min AHT, and {row['avg_routing']:.1f} routing days. ")
            if row['loop_rate'] > 0.2:
                desc_parts.append(f"Loop rate of {row['loop_rate']*100:.0f}% suggests cases are bouncing back to queues they already visited.")
            if is_anomaly:
                desc_parts.append(" These are the costliest cases in the system and the primary target for routing improvement.")

        narrative_items.append(html.Div([
            html.Div([html.Strong(cname), badge] if badge else [html.Strong(cname)],
                     style={'fontSize': '0.88rem', 'marginBottom': '0.2rem'}),
            html.P(''.join(desc_parts), style={'fontSize': '0.8rem', 'color': '#605E5C',
                                               'margin': '0 0 0.6rem 0', 'lineHeight': '1.5'}),
        ]))

    cluster_narrative = html.Div([
        html.Div("What the clusters reveal", style={
            'fontWeight': '700', 'fontSize': '0.9rem', 'color': '#201F1E',
            'marginBottom': '0.6rem',
        }),
        *narrative_items,
    ], style={
        'background': 'white', 'borderRadius': '8px', 'padding': '1.2rem 1.4rem',
        'boxShadow': '0 1.6px 3.6px 0 rgba(0,0,0,.132)', 'height': '100%',
    })

    model3_section = html.Div([
        html.Div([
            html.H6("Model 3: Clustering & Anomaly Detection", style={
                'fontWeight': '700', 'color': POWERBI_COLORS['secondary'], 'marginBottom': '0.3rem'}),
            html.P(f"KMeans (k={art3['best_k']}) — Silhouette Score: {art3['silhouette']:.3f}",
                   style={'fontSize': '0.85rem', 'color': '#555', 'marginBottom': '0'}),
            html.P([
                "Groups Messenger cases into natural journey archetypes based on transfer count, handle time, "
                "routing days, message volume, and queue visit patterns. Clusters with ",
                html.Strong("3+ average transfers are flagged as anomalies"),
                " because they represent cases that bounced far beyond normal routing. "
                f"Currently {anomaly_count} cases ({anomaly_pct:.1f}%) fall into anomaly clusters"
                + (f" ({', '.join(anomaly_clusters)}). " if anomaly_clusters else ". ")
                + "These are the cases worth investigating first: why did they bounce so many times, "
                "and could better initial routing have prevented it?",
            ], style={'fontSize': '0.8rem', 'color': '#888', 'marginBottom': '0'}),
        ], className="insight-card mb-3"),
        dbc.Row(profile_cards, className="mb-3"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_pca, config={'responsive': False})], md=7),
            dbc.Col([cluster_narrative], md=5),
        ]),
    ])

    return html.Div([
        guide_statement([
            "These models learn from your routing data to answer ",
            html.Strong("three questions humans struggle with at scale: "),
            "which cases are most likely to bounce, where should they have gone in the first place, and what ",
            "behavioural patterns keep repeating? The answers are predictions, not rules. ",
            html.Strong("Treat them as a second opinion."),
        ]),
        model1_section,
        html.Hr(className="divider"),
        model2_section,
        html.Hr(className="divider"),
        model3_section,
    ])


# ==================================
# RUN APP
# ==================================

if __name__ == '__main__':
    app.run(debug=False, port=8050)
