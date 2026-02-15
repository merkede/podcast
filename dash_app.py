"""
Executive Case Routing Analytics Dashboard - Dash Version
Built with Plotly Dash for better performance and interactivity
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==================================
# GENERATE SAMPLE DATA
# ==================================

def generate_sample_data():
    """Generate realistic sample data for dashboard testing"""
    np.random.seed(42)

    sample_rows = []
    case_id = 1000

    queue_names = [
        'General Enquiry', 'Technical Support', 'Billing', 'Payments',
        'Account Management', 'Renewals', 'Cancellations', 'Escalations',
        'VIP Support', 'Customer Service', 'Complaints', 'Refunds'
    ]

    # Generate diverse case patterns
    for _ in range(200):  # 200 cases
        case_id += 1

        # Determine case complexity
        complexity = np.random.choice(['simple', 'medium', 'complex', 'very_complex'],
                                     p=[0.4, 0.3, 0.2, 0.1])

        # Determine in-hours vs out-of-hours (70% in-hours)
        inhours = np.random.choice([0, 1], p=[0.3, 0.7])

        # Case journey based on complexity
        if complexity == 'simple':
            num_queues = 1
            base_aht = np.random.uniform(30, 120)
            base_days = np.random.uniform(0, 1)
            base_messages = np.random.randint(1, 3)
        elif complexity == 'medium':
            num_queues = 2
            base_aht = np.random.uniform(80, 200)
            base_days = np.random.uniform(1, 3)
            base_messages = np.random.randint(2, 6)
        elif complexity == 'complex':
            num_queues = np.random.choice([3, 4])
            base_aht = np.random.uniform(150, 350)
            base_days = np.random.uniform(2, 7)
            base_messages = np.random.randint(4, 10)
        else:  # very_complex
            num_queues = np.random.choice([4, 5, 6])
            base_aht = np.random.uniform(250, 500)
            base_days = np.random.uniform(5, 15)
            base_messages = np.random.randint(6, 15)

        # Out-of-hours penalty
        if inhours == 0:
            base_days *= 1.4
            base_aht *= 1.15
            base_messages = int(base_messages * 1.2)

        # Select queues for this case journey
        selected_queues = np.random.choice(queue_names, size=num_queues, replace=False).tolist()

        # 10% chance of a loop (revisiting a queue)
        has_loop = np.random.random() < 0.1
        if has_loop and num_queues > 1:
            loop_queue = selected_queues[np.random.randint(0, len(selected_queues))]
            selected_queues.append(loop_queue)

        # Generate timestamp
        created_at = pd.Timestamp('2025-10-01') + pd.Timedelta(days=np.random.randint(0, 60))

        total_days = 0
        cumulative_aht = 0

        # Create rows for each queue in journey
        for queue_order, queue_name in enumerate(selected_queues, 1):
            # Days in this queue
            if queue_order == len(selected_queues):  # Final queue
                days_in_queue = base_days * np.random.uniform(0.3, 0.7)
            else:  # Routing queues
                days_in_queue = base_days * np.random.uniform(0.1, 0.4) / max(1, num_queues - 1)

            total_days += days_in_queue

            # Process timestamp
            process_ts = created_at + pd.Timedelta(days=total_days)

            # AHT increases with each transfer
            queue_aht = base_aht * (1 + (queue_order - 1) * 0.15) / num_queues
            cumulative_aht += queue_aht

            # ASRT increases with complexity
            asrt = np.random.uniform(1, 5) * (1 + (queue_order - 1) * 0.3)

            # Close datetime
            close_datetime = created_at + pd.Timedelta(days=total_days) + pd.Timedelta(hours=np.random.randint(0, 24))
            close_date = close_datetime.strftime('%d/%m/%Y')
            close_time = close_datetime.strftime('%H:%M:%S.0')

            hours_to_close = (close_datetime - created_at).total_seconds() / 3600

            # Interactions increase with queue order
            interactions = base_messages + queue_order

            row = {
                'CASE_ID': case_id,
                'QUEUE_ORDER': queue_order,
                'QUEUE_NEW': queue_name,
                'PROCESS_TIMESTAMP': process_ts,
                'DAYS_IN_QUEUE': round(days_in_queue, 2),
                'CREATED_AT': created_at,
                'CLOSE_DATE': close_date,
                'CLOSE_TIME': close_time,
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
            }

            sample_rows.append(row)

    return pd.DataFrame(sample_rows)


# Generate data
df_raw = generate_sample_data()


# ==================================
# DATA PREPARATION
# ==================================

def prepare_data(df):
    """Prepare case-level aggregated data"""
    df = df.copy()
    df = df.sort_values(["CASE_ID", "QUEUE_ORDER"])

    # Case aggregates
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

    # Loop flag
    loops = (
        df.groupby("CASE_ID")["QUEUE_NEW"]
        .apply(lambda x: x.duplicated().any())
        .astype(int)
        .reset_index(name="loop_flag")
    )

    case = case.merge(loops, on="CASE_ID", how="left")

    # Derived metrics
    case['message_intensity'] = case['messages'] / (case['total_active_aht'] + 1)
    case['interaction_density'] = case['interactions'] / (case['total_active_aht'] + 1)
    case['ftr'] = (case['transfers'] == 0).astype(int)
    case['transfer_bin'] = pd.cut(case['transfers'],
                                   bins=[-0.1, 0, 1, 2, 100],
                                   labels=['0', '1', '2', '3+'])

    return df, case


df_raw, case_df = prepare_data(df_raw)

# Date range for filters
min_date = case_df['created_at'].min().date()
max_date = case_df['created_at'].max().date()

# ==================================
# INITIALIZE DASH APP
# ==================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)  # Hide "ID not found" errors
app.title = "Case Routing Analytics - Executive Dashboard"

# Power BI color palette
POWERBI_COLORS = {
    'primary': '#00BCF2',
    'secondary': '#742774',
    'success': '#00A86B',
    'warning': '#FFB900',
    'danger': '#E81123',
    'dark': '#252423',
    'light': '#F3F2F1'
}

# Chart color schemes (Power BI inspired)
CHART_COLORS = ['#00BCF2', '#742774', '#FFB900', '#E81123', '#00A86B',
                '#8764B8', '#F2C80F', '#0078D4', '#107C10', '#C50F1F']

# ==================================
# HELPER FUNCTIONS
# ==================================

def create_filter_section(tab_id):
    """Create filter controls for each tab - Power BI slicer style"""
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-filter me-2", style={'color': POWERBI_COLORS['primary']}),
                    html.Span("FILTERS", style={'fontWeight': '600', 'color': POWERBI_COLORS['primary'],
                                                 'fontSize': '0.9rem', 'letterSpacing': '0.5px'})
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("📅 Date Range", className="filter-label"),
                        dcc.DatePickerRange(
                            id=f'{tab_id}-date-filter',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date,
                            display_format='DD/MM/YYYY',
                            style={'width': '100%'}
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("🎯 Entry Queue", className="filter-label"),
                        dcc.Dropdown(
                            id=f'{tab_id}-queue-filter',
                            options=[{'label': q, 'value': q} for q in sorted(case_df.entry_queue.dropna().unique())],
                            value=sorted(case_df.entry_queue.dropna().unique()),
                            multi=True,
                            placeholder="Select entry queues...",
                            style={'fontSize': '0.9rem'}
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("⏰ Hours Type", className="filter-label"),
                        dcc.Dropdown(
                            id=f'{tab_id}-hours-filter',
                            options=[
                                {'label': '🌞 In Hours', 'value': 1},
                                {'label': '🌙 Out of Hours', 'value': 0}
                            ],
                            value=[0, 1],
                            multi=True,
                            placeholder="Select hours type...",
                            style={'fontSize': '0.9rem'}
                        ),
                    ], md=4),
                ]),
            ])
        ], className="filter-panel animated-card")
    ], className="mb-4")


def filter_data(case_data, start_date, end_date, queues, hours):
    """Filter case data based on selections"""
    if not queues or not hours:
        return pd.DataFrame()

    filtered = case_data[
        (case_data.created_at.dt.date >= pd.to_datetime(start_date).date()) &
        (case_data.created_at.dt.date <= pd.to_datetime(end_date).date()) &
        (case_data.entry_queue.isin(queues)) &
        (case_data.inhours.isin(hours))
    ]
    return filtered


def create_section_header(title, icon=""):
    """Create a styled section header"""
    return html.Div([
        html.H4([
            html.I(className=f"{icon} me-2", style={'color': POWERBI_COLORS['primary']}) if icon else None,
            title
        ], style={'color': '#201F1E', 'fontWeight': '600', 'marginBottom': '1rem'})
    ], className="section-header")


# ==================================
# LAYOUT
# ==================================

app.layout = dbc.Container([
    # Header Section
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("🎯 Executive Case Routing Analytics",
                           style={'color': POWERBI_COLORS['primary'], 'fontWeight': '700',
                                  'fontSize': '2.5rem', 'marginBottom': '0.5rem'}),
                    html.P("Board-Safe Insights: Separating Waiting, Working, and Customer Friction",
                          style={'color': '#605E5C', 'fontSize': '1.1rem', 'marginBottom': '0'}),
                    html.Hr(className="divider", style={'marginTop': '1rem'})
                ], className="text-center")
            ])
        ])
    ], className="mb-4"),

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='🎯 Executive Scorecard', value='tab-1'),
        dcc.Tab(label='🔁 Process & Routing', value='tab-2'),
        dcc.Tab(label='💰 Cost Inflation', value='tab-3'),
        dcc.Tab(label='😤 Customer Friction', value='tab-4'),
        dcc.Tab(label='⏰ In vs Out Hours', value='tab-5'),
        dcc.Tab(label='🔬 Queue Deep Dive', value='tab-6'),
        dcc.Tab(label='📊 Transfer Flow', value='tab-7'),
        dcc.Tab(label='🛤️ Journey Pathways', value='tab-8'),
    ]),

    html.Div(id='tabs-content', className="mt-4")

], fluid=True)


# ==================================
# CALLBACKS FOR TAB CONTENT
# ==================================

@callback(Output('tabs-content', 'children'),
          Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            create_filter_section('exec'),
            html.Div(id='exec-content')
        ])
    elif tab == 'tab-2':
        return html.Div([
            create_filter_section('process'),
            html.Div(id='process-content')
        ])
    elif tab == 'tab-3':
        return html.Div([
            create_filter_section('cost'),
            html.Div(id='cost-content')
        ])
    elif tab == 'tab-4':
        return html.Div([
            create_filter_section('friction'),
            html.Div(id='friction-content')
        ])
    elif tab == 'tab-5':
        return html.Div([
            create_filter_section('hours'),
            html.Div(id='hours-content')
        ])
    elif tab == 'tab-6':
        return html.Div([
            create_filter_section('deep'),
            html.Div(id='deep-content')
        ])
    elif tab == 'tab-7':
        return html.Div([
            create_filter_section('flow'),
            html.Div(id='flow-content')
        ])
    elif tab == 'tab-8':
        return html.Div([
            create_filter_section('journey'),
            html.Div(id='journey-content')
        ])


# ==================================
# TAB 1: EXECUTIVE SCORECARD
# ==================================

@callback(
    Output('exec-content', 'children'),
    [Input('exec-date-filter', 'start_date'),
     Input('exec-date-filter', 'end_date'),
     Input('exec-queue-filter', 'value'),
     Input('exec-hours-filter', 'value')]
)
def update_exec_tab(start_date, end_date, queues, hours):
    filtered = filter_data(case_df, start_date, end_date, queues, hours)

    if len(filtered) == 0:
        return html.Div("No data available for selected filters", className="alert alert-warning")

    # KPI Cards - Power BI Style with Gradients
    kpi_cards = dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("📊 Total Cases"),
                html.H2(f"{len(filtered):,}")
            ], className="kpi-card kpi-primary animated-card")
        ], md=2),
        dbc.Col([
            html.Div([
                html.H4("✅ First Touch Resolution"),
                html.H2(f"{filtered.ftr.mean()*100:.1f}%")
            ], className="kpi-card kpi-success animated-card")
        ], md=2),
        dbc.Col([
            html.Div([
                html.H4("🔄 Avg Transfers"),
                html.H2(f"{filtered.transfers.mean():.2f}")
            ], className="kpi-card kpi-warning animated-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.H4("⏱️ Median Routing Days"),
                html.H2(f"{filtered.routing_days.median():.1f}")
            ], className="kpi-card kpi-danger animated-card")
        ], md=2),
        dbc.Col([
            html.Div([
                html.H4("💼 Median AHT (min)"),
                html.H2(f"{filtered.total_active_aht.median():.0f}")
            ], className="kpi-card kpi-info animated-card")
        ], md=3),
    ], className="mb-4")

    # Three Dimensions
    waiting_data = [filtered.routing_days.sum(), filtered.final_queue_days.sum()]
    waiting_fig = go.Figure(data=[go.Pie(
        labels=['Routing (waste)', 'Final Queue (resolution)'],
        values=waiting_data,
        marker_colors=[POWERBI_COLORS['danger'], POWERBI_COLORS['primary']],
        hole=0.4,  # Donut chart
        textinfo='label+percent',
        textfont=dict(size=13, color='white'),
        hovertemplate='%{label}<br>%{value} days<br>%{percent}<extra></extra>'
    )])
    waiting_fig.update_layout(
        title=dict(text="WAITING: Calendar Days Distribution", font=dict(size=16, color='#201F1E', family='Segoe UI')),
        width=400,
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        autosize=False
    )

    # Working - AHT by transfers
    aht_by_transfer = filtered.groupby('transfer_bin')['total_active_aht'].median().reset_index()
    working_fig = go.Figure(data=[go.Bar(
        x=aht_by_transfer['transfer_bin'],
        y=aht_by_transfer['total_active_aht'],
        marker_color=POWERBI_COLORS['success'],
        marker_line=dict(color='white', width=2),
        text=aht_by_transfer['total_active_aht'].round(0),
        textposition='outside',
        textfont=dict(size=12, color='#201F1E'),
        hovertemplate='Transfers: %{x}<br>Median AHT: %{y:.0f} min<extra></extra>'
    )])
    working_fig.update_layout(
        title=dict(text="WORKING: AHT Inflation from Transfers", font=dict(size=16, color='#201F1E', family='Segoe UI')),
        xaxis_title="Number of Transfers",
        yaxis_title="Median Handle Time (min)",
        width=400,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        autosize=False
    )

    # Friction - Messages by transfers
    msg_by_transfer = filtered.groupby('transfer_bin')['messages'].median().reset_index()
    friction_fig = go.Figure(data=[go.Bar(
        x=msg_by_transfer['transfer_bin'],
        y=msg_by_transfer['messages'],
        marker_color=POWERBI_COLORS['warning'],
        marker_line=dict(color='white', width=2),
        text=msg_by_transfer['messages'].round(1),
        textposition='outside',
        textfont=dict(size=12, color='#201F1E'),
        hovertemplate='Transfers: %{x}<br>Median Messages: %{y:.1f}<extra></extra>'
    )])
    friction_fig.update_layout(
        title=dict(text="FRICTION: Customer Effort by Complexity", font=dict(size=16, color='#201F1E', family='Segoe UI')),
        xaxis_title="Number of Transfers",
        yaxis_title="Median Messages",
        width=400,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#EDEBE9'),
        autosize=False
    )

    # Transfer Impact Heatmap
    impact_data = filtered.groupby('transfer_bin').agg({
        'routing_days': 'median',
        'total_active_aht': 'median',
        'asrt': 'median',
        'messages': 'median',
        'close_hours': 'median'
    }).T

    if '0' in impact_data.columns:
        baseline = impact_data['0']
        impact_pct = impact_data.div(baseline, axis=0) * 100 - 100
    else:
        impact_pct = impact_data

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=impact_pct.values,
        x=impact_pct.columns,
        y=['Routing Days', 'Handle Time', 'ASRT', 'Messages', 'Total Hours'],
        colorscale=[[0, '#00A86B'], [0.5, '#FFFFFF'], [1, '#E81123']],  # Power BI green-white-red
        zmid=0,
        text=impact_pct.values.round(0),
        texttemplate='%{text:.0f}%',
        textfont={"size": 11, "color": "#201F1E", "family": "Segoe UI"},
        colorbar=dict(title="% Change<br>from FTR", titlefont=dict(family="Segoe UI")),
        hovertemplate='Metric: %{y}<br>Transfers: %{x}<br>Change: %{z:.0f}%<extra></extra>'
    ))
    heatmap_fig.update_layout(
        title=dict(text="Transfer Impact Matrix: % Increase vs First-Touch Resolution",
                  font=dict(size=16, color='#201F1E', family='Segoe UI')),
        xaxis_title="Number of Transfers",
        width=1000,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color='#201F1E'),
        xaxis=dict(showgrid=False, side='bottom'),
        yaxis=dict(showgrid=False),
        autosize=False
    )

    return html.Div([
        kpi_cards,
        html.Hr(),
        html.H4("The Three Dimensions: Waiting vs Working vs Friction", className="mb-3"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=waiting_fig, config={'responsive': False})], md=4),
            dbc.Col([dcc.Graph(figure=working_fig, config={'responsive': False})], md=4),
            dbc.Col([dcc.Graph(figure=friction_fig, config={'responsive': False})], md=4),
        ]),
        html.Hr(),
        dcc.Graph(figure=heatmap_fig, config={'responsive': False})
    ])


# ==================================
# TAB 2: PROCESS & ROUTING
# ==================================

@callback(
    Output('process-content', 'children'),
    [Input('process-date-filter', 'start_date'),
     Input('process-date-filter', 'end_date'),
     Input('process-queue-filter', 'value'),
     Input('process-hours-filter', 'value')]
)
def update_process_tab(start_date, end_date, queues, hours):
    filtered = filter_data(case_df, start_date, end_date, queues, hours)

    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    # Filter raw data
    filtered_cases = filtered.CASE_ID.unique()
    df_filtered = df_raw[df_raw.CASE_ID.isin(filtered_cases)]

    # Bottleneck Pareto
    queue_impact = (
        df_filtered.groupby("QUEUE_NEW")
        .agg(
            total_delay_days=("DAYS_IN_QUEUE", "sum"),
            median_days=("DAYS_IN_QUEUE", "median"),
            volume=("CASE_ID", "nunique")
        )
        .sort_values("total_delay_days", ascending=False)
        .head(10)
        .reset_index()
    )
    queue_impact['cumulative_pct'] = (queue_impact['total_delay_days'].cumsum() /
                                      queue_impact['total_delay_days'].sum() * 100)

    pareto_fig = make_subplots(specs=[[{"secondary_y": True}]])
    pareto_fig.add_trace(
        go.Bar(x=queue_impact['QUEUE_NEW'], y=queue_impact['total_delay_days'],
               name="Total Delay Days", marker_color='#e74c3c'),
        secondary_y=False
    )
    pareto_fig.add_trace(
        go.Scatter(x=queue_impact['QUEUE_NEW'], y=queue_impact['cumulative_pct'],
                   name="Cumulative %", mode='lines+markers', marker_color='#2c3e50'),
        secondary_y=True
    )
    pareto_fig.update_xaxes(title_text="Queue")
    pareto_fig.update_yaxes(title_text="Total Delay Days", secondary_y=False)
    pareto_fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
    pareto_fig.update_layout(title="Top 10 Bottleneck Queues (80/20 Rule)", width=550, height=500, autosize=False)

    # Entry Queue Effectiveness
    entry_perf = (
        filtered.groupby("entry_queue")
        .agg(
            cases=("CASE_ID", "count"),
            ftr_rate=("ftr", "mean"),
            avg_transfers=("transfers", "mean")
        )
        .sort_values("ftr_rate", ascending=True)
        .head(10)
        .reset_index()
    )

    entry_fig = go.Figure()
    entry_fig.add_trace(go.Bar(
        y=entry_perf['entry_queue'],
        x=entry_perf['ftr_rate'] * 100,
        orientation='h',
        name='FTR %',
        marker_color='#2ecc71'
    ))
    entry_fig.add_trace(go.Bar(
        y=entry_perf['entry_queue'],
        x=(1 - entry_perf['ftr_rate']) * 100,
        orientation='h',
        name='Transfer %',
        marker_color='#e74c3c'
    ))
    entry_fig.update_layout(
        title="Entry Queue FTR Performance",
        barmode='stack',
        xaxis_title="% of Cases",
        width=550,
        height=500,
        autosize=False
    )

    return html.Div([
        html.H4("Process & Routing Analysis"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=pareto_fig, config={'responsive': False})], md=6),
            dbc.Col([dcc.Graph(figure=entry_fig, config={'responsive': False})], md=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Loop/Rework Rate"),
                        html.H3(f"{filtered.loop_flag.mean()*100:.1f}%", className="text-danger")
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Cases with Rework"),
                        html.H3(f"{filtered.loop_flag.sum():,}")
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Multi-Transfer Cases"),
                        html.H3(f"{(filtered.transfers >= 2).mean()*100:.1f}%")
                    ])
                ])
            ], md=4),
        ])
    ])


# ==================================
# TAB 3: COST INFLATION
# ==================================

@callback(
    Output('cost-content', 'children'),
    [Input('cost-date-filter', 'start_date'),
     Input('cost-date-filter', 'end_date'),
     Input('cost-queue-filter', 'value'),
     Input('cost-hours-filter', 'value')]
)
def update_cost_tab(start_date, end_date, queues, hours):
    filtered = filter_data(case_df, start_date, end_date, queues, hours)

    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    # AHT by transfer count - Box plot
    box_fig = go.Figure()
    for transfer_bin in ['0', '1', '2', '3+']:
        data = filtered[filtered.transfer_bin == transfer_bin]['total_active_aht'].dropna()
        box_fig.add_trace(go.Box(
            y=data,
            name=transfer_bin,
            marker_color='#3498db'
        ))
    box_fig.update_layout(
        title="Handle Time Distribution by Transfer Count",
        xaxis_title="Number of Transfers",
        yaxis_title="Total Active Handle Time (min)",
        width=550,
        height=500,
        autosize=False
    )

    # Scatter: Routing Days vs AHT
    scatter_fig = px.scatter(
        filtered,
        x='routing_days',
        y='total_active_aht',
        color='transfers',
        color_continuous_scale='YlOrRd',
        title="Waiting vs Working: Different Problems",
        labels={'routing_days': 'Routing Days (Waiting)',
                'total_active_aht': 'Total Handle Time (Working)',
                'transfers': 'Transfers'}
    )
    scatter_fig.update_layout(width=550, height=500, autosize=False)

    return html.Div([
        html.H4("Cost Inflation Analysis"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=box_fig, config={'responsive': False})], md=6),
            dbc.Col([dcc.Graph(figure=scatter_fig, config={'responsive': False})], md=6),
        ])
    ])


# ==================================
# TAB 4: CUSTOMER FRICTION
# ==================================

@callback(
    Output('friction-content', 'children'),
    [Input('friction-date-filter', 'start_date'),
     Input('friction-date-filter', 'end_date'),
     Input('friction-queue-filter', 'value'),
     Input('friction-hours-filter', 'value')]
)
def update_friction_tab(start_date, end_date, queues, hours):
    filtered = filter_data(case_df, start_date, end_date, queues, hours)

    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    # Messages by transfer count
    msg_box = go.Figure()
    for transfer_bin in ['0', '1', '2', '3+']:
        data = filtered[filtered.transfer_bin == transfer_bin]['messages'].dropna()
        msg_box.add_trace(go.Box(
            y=data,
            name=transfer_bin,
            marker_color='#e67e22'
        ))
    msg_box.update_layout(
        title="Customer Effort Increases with Routing Friction",
        xaxis_title="Number of Transfers",
        yaxis_title="Messages Received from Customer",
        width=550,
        height=500,
        autosize=False
    )

    # Message Intensity Index
    intensity_data = filtered.groupby('transfer_bin')['message_intensity'].median().reset_index()
    intensity_fig = go.Figure(data=[go.Bar(
        x=intensity_data['transfer_bin'],
        y=intensity_data['message_intensity'],
        marker_color='#e67e22',
        text=intensity_data['message_intensity'].round(3),
        textposition='outside'
    )])
    intensity_fig.update_layout(
        title="Message Intensity Index (Messages per AHT Minute)",
        xaxis_title="Number of Transfers",
        yaxis_title="Messages / AHT Minute",
        width=550,
        height=500,
        autosize=False
    )

    return html.Div([
        html.H4("Customer Friction & Experience"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=msg_box, config={'responsive': False})], md=6),
            dbc.Col([dcc.Graph(figure=intensity_fig, config={'responsive': False})], md=6),
        ])
    ])


# ==================================
# TAB 5: IN VS OUT OF HOURS
# ==================================

@callback(
    Output('hours-content', 'children'),
    [Input('hours-date-filter', 'start_date'),
     Input('hours-date-filter', 'end_date'),
     Input('hours-queue-filter', 'value'),
     Input('hours-hours-filter', 'value')]
)
def update_hours_tab(start_date, end_date, queues, hours):
    filtered = filter_data(case_df, start_date, end_date, queues, hours)

    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    ih_data = filtered[filtered.inhours == 1]
    ooh_data = filtered[filtered.inhours == 0]

    if len(ooh_data) == 0:
        return html.Div("No out-of-hours data available", className="alert alert-warning")

    # Comparison metrics
    comparison_data = pd.DataFrame({
        'Metric': ['FTR Rate', 'Routing Days', 'Handle Time', 'ASRT', 'Messages'],
        'In Hours': [
            ih_data.ftr.mean() * 100,
            ih_data.routing_days.median(),
            ih_data.total_active_aht.median(),
            ih_data.asrt.median(),
            ih_data.messages.median()
        ],
        'Out of Hours': [
            ooh_data.ftr.mean() * 100,
            ooh_data.routing_days.median(),
            ooh_data.total_active_aht.median(),
            ooh_data.asrt.median(),
            ooh_data.messages.median()
        ]
    })
    comparison_data['% Difference'] = ((comparison_data['Out of Hours'] /
                                       comparison_data['In Hours'] - 1) * 100)

    # Side-by-side comparison
    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(
        y=comparison_data['Metric'],
        x=comparison_data['In Hours'],
        name='In Hours',
        orientation='h',
        marker_color='#2ecc71'
    ))
    compare_fig.add_trace(go.Bar(
        y=comparison_data['Metric'],
        x=comparison_data['Out of Hours'],
        name='Out of Hours',
        orientation='h',
        marker_color='#e74c3c'
    ))
    compare_fig.update_layout(
        title="Absolute Comparison: In Hours vs Out of Hours",
        barmode='group',
        width=550,
        height=500,
        autosize=False
    )

    # Penalty chart
    penalty_fig = go.Figure(data=[go.Bar(
        y=comparison_data['Metric'],
        x=comparison_data['% Difference'],
        orientation='h',
        marker_color=['#e74c3c' if v > 0 else '#2ecc71' for v in comparison_data['% Difference']],
        text=[f"{v:+.0f}%" for v in comparison_data['% Difference']],
        textposition='outside'
    )])
    penalty_fig.update_layout(
        title="Out-of-Hours Penalty (% Difference)",
        xaxis_title="% Difference (OOH vs IH)",
        width=550,
        height=500,
        autosize=False
    )
    penalty_fig.add_vline(x=0, line_color='black', line_width=1)

    return html.Div([
        html.H4("In-Hours vs Out-of-Hours Analysis"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=compare_fig, config={'responsive': False})], md=6),
            dbc.Col([dcc.Graph(figure=penalty_fig, config={'responsive': False})], md=6),
        ])
    ])


# ==================================
# TAB 6: QUEUE DEEP DIVE
# ==================================

@callback(
    Output('deep-content', 'children'),
    [Input('deep-date-filter', 'start_date'),
     Input('deep-date-filter', 'end_date'),
     Input('deep-queue-filter', 'value'),
     Input('deep-hours-filter', 'value')]
)
def update_deep_tab(start_date, end_date, queues, hours):
    filtered = filter_data(case_df, start_date, end_date, queues, hours)

    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    # Queue selector
    all_queues = sorted(df_raw.QUEUE_NEW.dropna().unique())

    return html.Div([
        html.H4("Queue Deep Dive"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Queue to Analyze:"),
                dcc.Dropdown(
                    id='queue-selector',
                    options=[{'label': q, 'value': q} for q in all_queues],
                    value=all_queues[0] if all_queues else None
                )
            ], md=6)
        ]),
        html.Div(id='queue-analysis')
    ])


@callback(
    Output('queue-analysis', 'children'),
    [Input('queue-selector', 'value'),
     Input('deep-date-filter', 'start_date'),
     Input('deep-date-filter', 'end_date'),
     Input('deep-queue-filter', 'value'),
     Input('deep-hours-filter', 'value')]
)
def update_queue_analysis(selected_queue, start_date, end_date, queues, hours):
    if not selected_queue:
        return html.Div()

    filtered = filter_data(case_df, start_date, end_date, queues, hours)
    subset_df = df_raw[df_raw.QUEUE_NEW == selected_queue]

    # Histogram of days in queue
    hist_fig = px.histogram(
        subset_df,
        x='DAYS_IN_QUEUE',
        nbins=30,
        title=f"Dwell Time Distribution: {selected_queue}"
    )
    hist_fig.add_vline(x=subset_df.DAYS_IN_QUEUE.median(),
                      line_dash="dash", line_color="red",
                      annotation_text=f"Median: {subset_df.DAYS_IN_QUEUE.median():.1f}")
    hist_fig.update_layout(width=1100, height=400, autosize=False)

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Cases Touching Queue"),
                    html.H3(f"{subset_df.CASE_ID.nunique():,}")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Median Days in Queue"),
                    html.H3(f"{subset_df.DAYS_IN_QUEUE.median():.1f}")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("P90 Days in Queue"),
                    html.H3(f"{subset_df.DAYS_IN_QUEUE.quantile(0.9):.1f}")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Delay Contribution"),
                    html.H3(f"{subset_df.DAYS_IN_QUEUE.sum():.0f} days")
                ])
            ])
        ], md=3),
        dbc.Col([dcc.Graph(figure=hist_fig, config={'responsive': False})], md=12, className="mt-3")
    ])


# ==================================
# TAB 7: TRANSFER FLOW
# ==================================

@callback(
    Output('flow-content', 'children'),
    [Input('flow-date-filter', 'start_date'),
     Input('flow-date-filter', 'end_date'),
     Input('flow-queue-filter', 'value'),
     Input('flow-hours-filter', 'value')]
)
def update_flow_tab(start_date, end_date, queues, hours):
    filtered = filter_data(case_df, start_date, end_date, queues, hours)

    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    filtered_cases = filtered.CASE_ID.unique()
    df_filtered = df_raw[df_raw.CASE_ID.isin(filtered_cases)]

    # Create transfer pairs
    transfer_flows = []
    for case_id in df_filtered.CASE_ID.unique():
        case_journey = df_filtered[df_filtered.CASE_ID == case_id].sort_values('QUEUE_ORDER')
        queues_list = case_journey.QUEUE_NEW.tolist()
        for i in range(len(queues_list) - 1):
            transfer_flows.append({
                'from_queue': queues_list[i],
                'to_queue': queues_list[i + 1]
            })

    if not transfer_flows:
        return html.Div("No transfer data available", className="alert alert-warning")

    transfer_df = pd.DataFrame(transfer_flows)
    top_paths = (
        transfer_df.groupby(['from_queue', 'to_queue'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(20)
    )
    top_paths['path'] = top_paths['from_queue'] + ' → ' + top_paths['to_queue']

    path_fig = go.Figure(data=[go.Bar(
        y=top_paths['path'],
        x=top_paths['count'],
        orientation='h',
        marker_color='#3498db',
        text=top_paths['count'],
        textposition='outside'
    )])
    path_fig.update_layout(
        title="Top 20 Transfer Paths",
        xaxis_title="Number of Transfers",
        width=1100,
        height=700,
        autosize=False
    )
    path_fig.update_yaxes(autorange="reversed")

    return html.Div([
        html.H4("Transfer Flow Analysis"),
        dcc.Graph(figure=path_fig, config={'responsive': False})
    ])


# ==================================
# TAB 8: JOURNEY PATHWAYS
# ==================================

@callback(
    Output('journey-content', 'children'),
    [Input('journey-date-filter', 'start_date'),
     Input('journey-date-filter', 'end_date'),
     Input('journey-queue-filter', 'value'),
     Input('journey-hours-filter', 'value')]
)
def update_journey_tab(start_date, end_date, queues, hours):
    filtered = filter_data(case_df, start_date, end_date, queues, hours)

    if len(filtered) == 0:
        return html.Div("No data available", className="alert alert-warning")

    # Queue selector
    all_queues = sorted(df_raw.QUEUE_NEW.dropna().unique())

    return html.Div([
        html.H4("Customer Journey Pathways Analysis"),
        html.P("Visualize how customers flow through queues - see both forward paths (where they go) and backward paths (how they arrived)",
               className="text-muted mb-4"),

        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Queue to Analyze:", className="fw-bold"),
                        dcc.Dropdown(
                            id='journey-queue-selector',
                            options=[{'label': q, 'value': q} for q in all_queues],
                            value=all_queues[0] if all_queues else None,
                            placeholder="Choose a queue..."
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("Journey Depth (Levels):", className="fw-bold"),
                        dcc.Slider(
                            id='journey-depth-slider',
                            min=2,
                            max=5,
                            value=3,
                            marks={i: str(i) for i in range(2, 6)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=6),
                ])
            ])
        ], className="mb-4"),

        html.Div(id='journey-analysis')
    ])


@callback(
    Output('journey-analysis', 'children'),
    [Input('journey-queue-selector', 'value'),
     Input('journey-depth-slider', 'value'),
     Input('journey-date-filter', 'start_date'),
     Input('journey-date-filter', 'end_date'),
     Input('journey-queue-filter', 'value'),
     Input('journey-hours-filter', 'value')]
)
def update_journey_analysis(selected_queue, depth, start_date, end_date, queues, hours):
    if not selected_queue:
        return html.Div()

    filtered = filter_data(case_df, start_date, end_date, queues, hours)
    filtered_cases = filtered.CASE_ID.unique()
    df_filtered = df_raw[df_raw.CASE_ID.isin(filtered_cases)]

    # ==================================
    # FORWARD VIEW: Starting from selected queue
    # ==================================

    forward_cases = df_filtered[df_filtered.QUEUE_NEW == selected_queue].CASE_ID.unique()
    forward_journeys = df_filtered[df_filtered.CASE_ID.isin(forward_cases)].sort_values(['CASE_ID', 'QUEUE_ORDER'])

    # Build forward paths
    forward_paths = []
    for case_id in forward_cases:
        journey = forward_journeys[forward_journeys.CASE_ID == case_id].QUEUE_NEW.tolist()
        if selected_queue in journey:
            start_idx = journey.index(selected_queue)
            path = journey[start_idx:start_idx + depth]
            if len(path) > 1:  # Only include if there's a next step
                forward_paths.append(path)

    # Create forward Sankey
    forward_sankey = create_sankey_from_paths(forward_paths, f"Forward Journey from {selected_queue}")

    # ==================================
    # BACKWARD VIEW: Ending at selected queue
    # ==================================

    backward_cases = df_filtered[df_filtered.QUEUE_NEW == selected_queue].CASE_ID.unique()
    backward_journeys = df_filtered[df_filtered.CASE_ID.isin(backward_cases)].sort_values(['CASE_ID', 'QUEUE_ORDER'])

    # Build backward paths
    backward_paths = []
    for case_id in backward_cases:
        journey = backward_journeys[backward_journeys.CASE_ID == case_id].QUEUE_NEW.tolist()
        if selected_queue in journey:
            end_idx = journey.index(selected_queue)
            start_idx = max(0, end_idx - depth + 1)
            path = journey[start_idx:end_idx + 1]
            if len(path) > 1:  # Only include if there's a previous step
                backward_paths.append(path)

    # Create backward Sankey
    backward_sankey = create_sankey_from_paths(backward_paths, f"Backward Journey to {selected_queue}")

    # ==================================
    # TOP COMPLETE PATHS
    # ==================================

    # Get complete journeys through this queue
    complete_paths = []
    for case_id in forward_cases:
        journey = df_filtered[df_filtered.CASE_ID == case_id].sort_values('QUEUE_ORDER').QUEUE_NEW.tolist()
        complete_paths.append(' → '.join(journey))

    path_counts = pd.Series(complete_paths).value_counts().head(10).reset_index()
    path_counts.columns = ['Journey Path', 'Number of Cases']

    # Path table
    path_table = dbc.Table.from_dataframe(
        path_counts,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="mt-3"
    )

    # ==================================
    # STATISTICS CARDS
    # ==================================

    stats_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Cases Through This Queue"),
                    html.H3(f"{len(forward_cases):,}", className="text-primary")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Unique Forward Paths"),
                    html.H3(f"{len(set(map(tuple, forward_paths)))}", className="text-success")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Unique Backward Paths"),
                    html.H3(f"{len(set(map(tuple, backward_paths)))}", className="text-warning")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Avg Journey Length"),
                    html.H3(f"{np.mean([len(p) for p in complete_paths]) if complete_paths else 0:.1f}",
                           className="text-info")
                ])
            ])
        ], md=3),
    ], className="mb-4")

    return html.Div([
        stats_cards,

        html.Hr(),

        # Forward Journey
        html.H5(f"📤 Forward View: Where do customers go FROM {selected_queue}?", className="mb-3"),
        html.P("This shows the paths customers take AFTER entering this queue", className="text-muted"),
        dcc.Graph(figure=forward_sankey, config={'responsive': False}),

        html.Hr(),

        # Backward Journey
        html.H5(f"📥 Backward View: How do customers arrive TO {selected_queue}?", className="mb-3"),
        html.P("This shows the paths customers took BEFORE reaching this queue", className="text-muted"),
        dcc.Graph(figure=backward_sankey, config={'responsive': False}),

        html.Hr(),

        # Complete Paths
        html.H5(f"🛤️ Top 10 Complete Journey Paths Through {selected_queue}", className="mb-3"),
        html.P("Full end-to-end customer journeys that include this queue", className="text-muted"),
        path_table
    ])


def create_sankey_from_paths(paths, title):
    """Create a Sankey diagram from a list of journey paths"""

    if not paths:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No journey data available for this selection",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig

    # Build source-target pairs with counts
    links = []
    for path in paths:
        for i in range(len(path) - 1):
            source = f"{path[i]} (Step {i+1})"
            target = f"{path[i+1]} (Step {i+2})"
            links.append((source, target))

    # Count occurrences
    link_counts = pd.Series(links).value_counts().reset_index()
    link_counts.columns = ['link', 'count']
    link_counts[['source', 'target']] = pd.DataFrame(link_counts['link'].tolist(), index=link_counts.index)

    # Create node list
    all_nodes = list(set(link_counts['source'].tolist() + link_counts['target'].tolist()))
    node_dict = {node: idx for idx, node in enumerate(all_nodes)}

    # Map to indices
    source_indices = [node_dict[s] for s in link_counts['source']]
    target_indices = [node_dict[t] for t in link_counts['target']]
    values = link_counts['count'].tolist()

    # Create color palette
    colors = px.colors.qualitative.Set3
    node_colors = [colors[i % len(colors)] for i in range(len(all_nodes))]

    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            label=[f"{v} cases" for v in values]
        )
    )])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='#2c3e50')
        ),
        font=dict(size=12),
        width=1100,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        autosize=False
    )

    return fig


# ==================================
# RUN APP
# ==================================

if __name__ == '__main__':
    app.run(debug=True, port=8050)
