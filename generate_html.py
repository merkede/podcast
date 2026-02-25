#!/usr/bin/env python3
"""
generate_html.py — Static HTML dashboard for Messenger Transfer Analytics.

Usage:
    python generate_html.py --data your_data.csv
    python generate_html.py --data your_data.csv --out ./dist

Outputs to the output directory:
    index.html           — The full dashboard app
    cases.parquet        — Case-level data
    transitions.parquet  — Queue transition data for journey/routing views

Deploy all three files to Azure Static Web Apps or Azure Blob with static hosting.

Local testing (DuckDB-WASM needs HTTP, not file://):
    cd dist && python3 -m http.server 8080
    Open: http://localhost:8080
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION  (mirrors dash_app.py)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(df):
    df = df.copy()
    df = df.sort_values(["CASE_ID", "QUEUE_ORDER"])

    case = (
        df.groupby("CASE_ID")
        .agg(
            transfers=("QUEUE_ORDER", lambda x: x.max() - 1),
            queues_touched=("QUEUE_ORDER", "max"),
            routing_days=("DAYS_IN_QUEUE", lambda x: x.iloc[:-1].sum() if len(x) > 1 else 0),
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

    # Seconds → minutes
    case["total_active_aht"] = case["total_active_aht"] / 60

    case["message_intensity"] = case["messages"] / (case["total_active_aht"].fillna(0) + 1)
    case["ftr"] = (case["transfers"].fillna(0) == 0).astype(int)
    case["transfer_bin"] = pd.cut(
        case["transfers"].fillna(0), bins=[-0.1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"]
    )
    eq = case["entry_queue"].fillna("")
    case["segment"] = np.where(
        eq.str.startswith("HD RTL A") & ~eq.str.startswith("HD RTL A PRT"),
        "Retail", "Claims",
    )
    for col in ["created_at", "close_datetime"]:
        if case[col].dtype == object:
            case[col] = pd.to_datetime(case[col], errors="coerce")

    # Derived time columns used by the heatmap tab
    case["day_of_week"] = pd.to_datetime(case["created_at"], errors="coerce").dt.dayofweek
    case["hour_of_day"] = pd.to_datetime(case["close_datetime"], errors="coerce").dt.hour
    case["day_of_week"] = case["day_of_week"].fillna(0).astype(int)
    case["hour_of_day"] = case["hour_of_day"].fillna(12).astype(int)

    return df, case


# ══════════════════════════════════════════════════════════════════════════════
# PARQUET EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_cases(case_df, path):
    df = case_df.copy()
    df["CASE_ID"] = df["CASE_ID"].astype(str)
    df["transfer_bin"] = df["transfer_bin"].astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["close_datetime"] = pd.to_datetime(df["close_datetime"], errors="coerce")
    df["month"] = df["created_at"].dt.strftime("%Y-%m")
    df["created_date"] = df["created_at"].dt.strftime("%Y-%m-%d")
    # Drop heavy columns not needed in browser
    drop_cols = [c for c in ["asrt", "interaction_density", "message_intensity", "ftr",
                              "queues_touched", "close_hours"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")
    mb = path.stat().st_size / 1024 / 1024
    print(f"  cases.parquet      {len(df):>10,} rows  {mb:.1f} MB")


def export_transitions(df_raw, path):
    cols = [c for c in ["CASE_ID", "QUEUE_NEW", "QUEUE_ORDER", "DAYS_IN_QUEUE"] if c in df_raw.columns]
    df = df_raw[cols].copy()
    df["CASE_ID"] = df["CASE_ID"].astype(str)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")
    mb = path.stat().st_size / 1024 / 1024
    print(f"  transitions.parquet{len(df):>10,} rows  {mb:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_html(case_df, min_date, max_date, all_queues):
    # Pre-compute overview KPIs from full data for initial display
    total = len(case_df)
    drr = (case_df["transfers"] == 0).mean() * 100
    avg_xfer = case_df["transfers"].mean()
    med_aht = case_df["total_active_aht"].median()
    multi_rate = (case_df["transfers"] >= 2).mean() * 100
    loop_rate = case_df["loop_flag"].mean() * 100

    # Escape for JSON embedding
    queues_json = json.dumps(all_queues)

    months = sorted(case_df["created_at"].dropna().dt.strftime("%Y-%m").unique().tolist())
    months_json = json.dumps(months)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Messenger Transfer Analytics — Hastings Direct</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{--hd-green:#217346;--pbi-blue:#0078D4;--pbi-danger:#E81123;--pbi-warning:#FFB900;
  --pbi-success:#107C10;--pbi-purple:#742774;--pbi-teal:#00BCF2;}}
body{{font-family:'Segoe UI',sans-serif;background:#F0F2F5;color:#201F1E;font-size:.9rem;}}
/* ── Header ── */
.hd-header{{background:#1a1a2e;padding:.6rem 1.4rem;display:flex;align-items:center;gap:1rem;
  box-shadow:0 2px 8px rgba(0,0,0,.4);position:sticky;top:0;z-index:1000;}}
.hd-logo{{background:var(--hd-green);color:#fff;font-weight:800;font-size:1.1rem;padding:.3rem .8rem;
  border-radius:4px;letter-spacing:.5px;}}
.hd-title{{color:#fff;font-size:.95rem;font-weight:600;opacity:.9;}}
/* ── Filters ── */
.filter-panel{{background:#fff;padding:1rem 1.4rem;border-bottom:1px solid #E1DFDD;
  box-shadow:0 1px 4px rgba(0,0,0,.08);}}
.filter-label{{font-size:.65rem;font-weight:700;color:#444;text-transform:uppercase;
  letter-spacing:.5px;margin-bottom:.25rem;}}
.filter-panel select,.filter-panel input{{font-size:.82rem;border:1px solid #C8C6C4;border-radius:4px;
  padding:.3rem .5rem;width:100%;}}
.filter-panel select[multiple]{{height:80px;}}
/* ── Tabs ── */
.tab-nav{{background:#fff;border-bottom:2px solid #E1DFDD;padding:0 1rem;
  position:sticky;top:53px;z-index:999;display:flex;gap:0;overflow-x:auto;}}
.tab-btn{{border:none;background:none;padding:.65rem 1.1rem;font-size:.82rem;font-weight:600;
  color:#605E5C;cursor:pointer;border-bottom:3px solid transparent;white-space:nowrap;
  transition:all .15s;}}
.tab-btn.active{{color:var(--pbi-blue);border-bottom-color:var(--pbi-blue);}}
.tab-btn:hover:not(.active){{color:#201F1E;background:#F3F2F1;}}
/* ── Content ── */
.content-area{{padding:1.2rem 1.4rem;max-width:1400px;margin:0 auto;}}
.tab-panel{{display:none;}}.tab-panel.active{{display:block;}}
/* ── Cards ── */
.kpi-card{{background:#fff;border-radius:8px;padding:.9rem 1.1rem;
  box-shadow:0 1.6px 3.6px rgba(0,0,0,.132);margin-bottom:1rem;}}
.kpi-card h4{{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.4px;
  color:#605E5C;margin:0 0 .3rem;}}
.kpi-card h2{{font-size:1.6rem;font-weight:700;margin:0;}}
.kpi-primary h2{{color:var(--pbi-blue);}}.kpi-danger h2{{color:var(--pbi-danger);}}
.kpi-success h2{{color:var(--pbi-success);}}.kpi-warning h2{{color:#B8860B;}}
.kpi-info h2{{color:var(--pbi-purple);}}
/* ── Chart containers ── */
.chart-card{{background:#fff;border-radius:8px;padding:1rem;
  box-shadow:0 1.6px 3.6px rgba(0,0,0,.132);margin-bottom:1rem;}}
/* ── Guide statement ── */
.guide-stmt{{background:#F3F2F1;border-left:3px solid var(--pbi-blue);border-radius:0 6px 6px 0;
  padding:.8rem 1.2rem;margin-bottom:1.2rem;font-size:.88rem;color:#444;font-style:italic;
  line-height:1.6;}}
/* ── Toggle buttons ── */
.toggle-group{{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.8rem;}}
.toggle-btn{{border:1px solid var(--pbi-blue);border-radius:4px;padding:.3rem .8rem;
  font-size:.78rem;font-weight:600;color:var(--pbi-blue);background:#fff;cursor:pointer;}}
.toggle-btn.active{{background:var(--pbi-blue);color:#fff;}}
/* ── Table ── */
.data-table{{font-size:.78rem;}}
.data-table th{{background:var(--pbi-blue);color:#fff;font-weight:700;font-size:.7rem;
  text-transform:uppercase;letter-spacing:.3px;padding:8px 10px;white-space:nowrap;}}
.data-table td{{padding:6px 10px;border-bottom:1px solid #F3F2F1;}}
.data-table tbody tr:hover{{background:#EFF6FF;cursor:pointer;}}
/* ── Anomaly badge ── */
.anomaly-badge{{background:var(--pbi-danger);color:#fff;font-size:.6rem;font-weight:700;
  padding:1px 5px;border-radius:3px;margin-left:.3rem;vertical-align:middle;}}
/* ── Loading overlay ── */
#loading-overlay{{position:fixed;inset:0;background:rgba(255,255,255,.92);z-index:9999;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;}}
.spinner{{width:48px;height:48px;border:5px solid #E1DFDD;border-top-color:var(--pbi-blue);
  border-radius:50%;animation:spin 1s linear infinite;}}
@keyframes spin{{to{{transform:rotate(360deg);}}}}
/* ── Pagination ── */
.pager{{display:flex;gap:.3rem;align-items:center;margin-top:.5rem;}}
.pager button{{border:1px solid #C8C6C4;background:#fff;border-radius:4px;padding:.25rem .6rem;
  font-size:.78rem;cursor:pointer;}}
.pager button.active{{background:var(--pbi-blue);color:#fff;border-color:var(--pbi-blue);}}
/* ── Insight card ── */
.insight-card{{background:#EFF6FF;border:1px solid #BDD7F0;border-radius:6px;
  padding:.7rem 1rem;margin-bottom:.8rem;font-size:.87rem;color:#333;}}
</style>
</head>
<body>

<!-- LOADING OVERLAY -->
<div id="loading-overlay">
  <div class="spinner"></div>
  <div style="font-size:.95rem;font-weight:600;color:#201F1E;">Loading analytics engine...</div>
  <div id="loading-status" style="font-size:.8rem;color:#888;">Initialising DuckDB</div>
</div>

<!-- HEADER -->
<div class="hd-header">
  <div class="hd-logo">H</div>
  <div class="hd-title">Messenger Transfer Analytics &nbsp;|&nbsp; Hastings Direct</div>
  <div style="margin-left:auto;color:#aaa;font-size:.75rem;" id="case-count-badge"></div>
</div>

<!-- FILTERS -->
<div class="filter-panel">
  <div class="row g-2">
    <div class="col-md-3">
      <div class="filter-label">Date Range</div>
      <div class="d-flex gap-1">
        <input type="date" id="f-start" value="{min_date}">
        <input type="date" id="f-end" value="{max_date}">
      </div>
    </div>
    <div class="col-md-3">
      <div class="filter-label">Entry Queue</div>
      <select id="f-queue" multiple title="All queues (select to filter)"></select>
    </div>
    <div class="col-md-2">
      <div class="filter-label">Hours</div>
      <select id="f-hours" multiple>
        <option value="1" selected>In-Hours</option>
        <option value="0" selected>Out-of-Hours</option>
      </select>
    </div>
    <div class="col-md-2">
      <div class="filter-label">Segment</div>
      <select id="f-segment" multiple>
        <option value="Retail" selected>Retail</option>
        <option value="Claims" selected>Claims</option>
      </select>
    </div>
    <div class="col-md-2 d-flex align-items-end">
      <button onclick="applyFilters()" class="btn btn-sm btn-primary w-100">Apply Filters</button>
    </div>
  </div>
</div>

<!-- TABS -->
<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
  <button class="tab-btn" onclick="switchTab('process',this)">Process &amp; Routing</button>
  <button class="tab-btn" onclick="switchTab('cost',this)">Cost &amp; Effort</button>
  <button class="tab-btn" onclick="switchTab('hours',this)">Hours &amp; Transfer</button>
  <button class="tab-btn" onclick="switchTab('queue',this)">Queue Intelligence</button>
  <button class="tab-btn" onclick="switchTab('journey',this)">Journey Pathways</button>
  <button class="tab-btn" onclick="switchTab('explorer',this)">Data Explorer</button>
</div>

<!-- TAB CONTENT -->
<div class="content-area">

  <!-- ── TAB 1: OVERVIEW ── -->
  <div id="tab-overview" class="tab-panel active">
    <div class="guide-stmt">
      <strong>This report quantifies the cost of mis-routing in Messenger.</strong>
      Every transfer that could have been avoided represents wasted agent time, customer frustration,
      and compounding operational cost. Use the filters above to slice by date, queue, hours, and segment.
    </div>
    <div class="row g-3" id="overview-kpis"></div>
    <div class="row g-3 mt-1">
      <div class="col-md-6"><div class="chart-card"><div id="chart-ov-transfers"></div></div></div>
      <div class="col-md-6"><div class="chart-card"><div id="chart-ov-segment"></div></div></div>
    </div>
  </div>

  <!-- ── TAB 2: PROCESS & ROUTING ── -->
  <div id="tab-process" class="tab-panel">
    <div class="guide-stmt">
      <strong>Not all queues add value, some just add delay.</strong>
      The intermediary queues shown here are where Messenger cases sit waiting between handoffs,
      contributing nothing to resolution. If a queue appears frequently in the Pareto,
      it's either a structural bottleneck or a sign that cases are being sent there by mistake.
    </div>
    <div class="row g-3" id="process-kpis"></div>
    <div class="row g-3 mt-1">
      <div class="col-md-6"><div class="chart-card"><div id="chart-pareto"></div></div></div>
      <div class="col-md-6"><div class="chart-card"><div id="chart-entry-dist"></div></div></div>
    </div>
  </div>

  <!-- ── TAB 3: COST & EFFORT ── -->
  <div id="tab-cost" class="tab-panel">
    <div class="guide-stmt" id="cost-guide-stmt">Loading...</div>
    <div class="row g-3" id="cost-kpis"></div>
    <div id="cost-insight" class="insight-card"></div>
    <p style="font-size:.78rem;color:#999;">Click any box to see the individual cases in that group.</p>
    <div class="row g-3">
      <div class="col-md-6"><div class="chart-card"><div id="chart-aht-box"></div></div></div>
      <div class="col-md-6"><div class="chart-card"><div id="chart-msg-box"></div></div></div>
    </div>
    <div class="chart-card mt-2"><div id="chart-multiplier"></div></div>
  </div>

  <!-- ── TAB 4: HOURS & TRANSFER ── -->
  <div id="tab-hours" class="tab-panel">
    <div class="guide-stmt" id="hours-guide-stmt">Loading...</div>
    <div class="row g-3" id="hours-kpis"></div>
    <div class="toggle-group" id="heatmap-toggles">
      <button class="toggle-btn active" onclick="setHeatmapView('volume',this)">Transfer Volume</button>
      <button class="toggle-btn" onclick="setHeatmapView('aht',this)">Median AHT</button>
      <button class="toggle-btn" onclick="setHeatmapView('messages',this)">Customer Messages</button>
      <button class="toggle-btn" onclick="setHeatmapView('routing',this)">Routing Wait</button>
      <button class="toggle-btn" onclick="setHeatmapView('inhours',this)">In/Out Hours</button>
    </div>
    <div class="chart-card"><div id="chart-heatmap"></div></div>
  </div>

  <!-- ── TAB 5: QUEUE INTELLIGENCE ── -->
  <div id="tab-queue" class="tab-panel">
    <div class="guide-stmt">
      <strong>Every queue tells a story: is it resolving cases, or just passing them along?</strong>
      Select a queue to see who sends it work, where it sends cases next, and how long they dwell.
      If a queue has high inbound volume but low resolution, it's acting as an expensive middleman.
    </div>
    <div class="row g-2 mb-3">
      <div class="col-md-4">
        <div class="filter-label">Select Queue</div>
        <select id="qi-queue-select" class="form-select form-select-sm" onchange="renderQueueIntel()"></select>
      </div>
    </div>
    <div class="row g-3" id="qi-kpis"></div>
    <div class="row g-3 mt-1">
      <div class="col-md-6"><div class="chart-card"><div id="chart-qi-inbound"></div></div></div>
      <div class="col-md-6"><div class="chart-card"><div id="chart-qi-outbound"></div></div></div>
    </div>
  </div>

  <!-- ── TAB 6: JOURNEY PATHWAYS ── -->
  <div id="tab-journey" class="tab-panel">
    <div class="guide-stmt">
      <strong>The shortest path to resolution is the cheapest one.</strong>
      This tab maps how Messenger cases actually flow through the business.
      Every extra hop on the journey is time, effort, and customer patience burned.
    </div>
    <div class="row g-2 mb-3">
      <div class="col-md-4">
        <div class="filter-label">Select Queue to Analyse</div>
        <select id="journey-queue-select" class="form-select form-select-sm" onchange="renderJourney()"></select>
      </div>
      <div class="col-md-2">
        <div class="filter-label">Depth</div>
        <select id="journey-depth" class="form-select form-select-sm" onchange="renderJourney()">
          <option value="2">2</option><option value="3" selected>3</option>
          <option value="4">4</option><option value="5">5</option>
        </select>
      </div>
    </div>
    <div class="row g-3" id="journey-kpis"></div>
    <div id="journey-avoidable" class="insight-card" style="display:none;"></div>
    <div class="chart-card mt-2"><div id="chart-sankey-fwd"></div></div>
    <div class="chart-card mt-2"><div id="chart-sankey-bwd"></div></div>
    <div class="mt-2">
      <h6 style="font-weight:700;">Top 10 Complete Paths <small class="text-muted">(click row for case detail)</small></h6>
      <div id="journey-path-table"></div>
    </div>
  </div>

  <!-- ── TAB 7: DATA EXPLORER ── -->
  <div id="tab-explorer" class="tab-panel">
    <div class="guide-stmt">
      <strong>Everything in this report is built from the data below.</strong>
      Browse case-level summaries, filter by transfer count, then download the CSV.
      No black boxes.
    </div>
    <div class="d-flex gap-2 align-items-center mb-2">
      <div class="filter-label mb-0">Transfer Count:</div>
      <select id="ex-xfer" class="form-select form-select-sm" style="width:auto;" onchange="renderExplorer()">
        <option value="all" selected>All</option>
        <option value="0">0 — Direct Resolution</option>
        <option value="1">1 Transfer</option>
        <option value="2">2 Transfers</option>
        <option value="3+">3+ Transfers</option>
      </select>
      <button class="btn btn-sm btn-outline-primary" onclick="downloadCSV()">Download CSV</button>
      <span id="ex-count" class="text-muted" style="font-size:.78rem;"></span>
    </div>
    <div id="explorer-table-container"></div>
    <div class="pager" id="explorer-pager"></div>
  </div>

</div><!-- /content-area -->

<!-- CASE DETAIL MODAL -->
<div class="modal fade" id="caseModal" tabindex="-1">
  <div class="modal-dialog modal-xl">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="caseModalTitle"></h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body" id="caseModalBody"></div>
    </div>
  </div>
</div>

<script type="module">
// ═══════════════════════════════════════════════════════
// CONSTANTS (pre-computed by Python)
// ═══════════════════════════════════════════════════════
const MIN_DATE = '{min_date}';
const MAX_DATE = '{max_date}';
const ALL_QUEUES = {queues_json};
const MONTHS = {months_json};

const DAY_NAMES = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
const HOUR_LABELS = Array.from({{length:24}}, (_,i) => String(i).padStart(2,'0')+':00');

const COLORS = {{
  primary:'#0078D4', danger:'#E81123', success:'#107C10',
  warning:'#E8820C', purple:'#742774', teal:'#00BCF2',
}};
const BIN_COLORS = {{'0':COLORS.success,'1':COLORS.warning,'2':COLORS.warning,'3+':COLORS.danger}};
const RED_SCALE = [[0,'#FFF5F5'],[0.2,'#FFCDD2'],[0.4,'#EF9A9A'],
                   [0.6,'#E57373'],[0.8,'#D32F2F'],[1,'#8B0000']];
const CHART_COLORS = ['#0078D4','#FFB900','#E81123','#107C10','#742774','#00BCF2','#E8820C'];

// ═══════════════════════════════════════════════════════
// DUCKDB INIT
// ═══════════════════════════════════════════════════════
import * as duckdb from 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm/+esm';

let CONN = null;
let EXPLORER_DATA = [];
let EXPLORER_PAGE = 0;
const PAGE_SIZE = 50;
let HEATMAP_VIEW = 'volume';
let HEATMAP_DATA = null;
let JOURNEY_PATHS = {{}};  // path_str -> [case_ids]

async function initDB() {{
  setStatus('Loading DuckDB engine...');
  const bundles = duckdb.getJsDelivrBundles();
  const bundle = await duckdb.selectBundle(bundles);
  const workerUrl = URL.createObjectURL(
    new Blob([`importScripts("${{bundle.mainWorker}}");`], {{type:'text/javascript'}})
  );
  const worker = new Worker(workerUrl);
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  CONN = await db.connect();

  setStatus('Loading case data...');
  const base = new URL('.', window.location.href).href;
  await db.registerFileURL('cases.parquet', base + 'cases.parquet', duckdb.DuckDBDataProtocol.HTTP, false);
  await db.registerFileURL('transitions.parquet', base + 'transitions.parquet', duckdb.DuckDBDataProtocol.HTTP, false);
  await CONN.query("CREATE OR REPLACE VIEW cases AS SELECT * FROM read_parquet('cases.parquet')");
  await CONN.query("CREATE OR REPLACE VIEW transitions AS SELECT * FROM read_parquet('transitions.parquet')");

  setStatus('Populating filters...');
  populateQueueFilter();
  populateQIQueues();

  setStatus('Rendering dashboard...');
  await applyFilters();

  document.getElementById('loading-overlay').style.display = 'none';
}}

function setStatus(msg) {{
  document.getElementById('loading-status').textContent = msg;
}}

// ═══════════════════════════════════════════════════════
// FILTER STATE
// ═══════════════════════════════════════════════════════
function getFilterState() {{
  const start = document.getElementById('f-start').value || MIN_DATE;
  const end   = document.getElementById('f-end').value || MAX_DATE;
  const qSel  = document.getElementById('f-queue');
  const hSel  = document.getElementById('f-hours');
  const sSel  = document.getElementById('f-segment');
  const queues   = Array.from(qSel.selectedOptions).map(o => o.value);
  const hours    = Array.from(hSel.selectedOptions).map(o => o.value);
  const segments = Array.from(sSel.selectedOptions).map(o => o.value);
  return {{start, end, queues, hours, segments}};
}}

function buildWhere(f, alias) {{
  const a = alias ? alias + '.' : '';
  const parts = [
    `CAST(${{a}}created_at AS DATE) >= '${{f.start}}'`,
    `CAST(${{a}}created_at AS DATE) <= '${{f.end}}'`,
  ];
  if (f.segments.length === 1) parts.push(`${{a}}segment = '${{f.segments[0]}}'`);
  if (f.hours.length === 1) parts.push(`${{a}}inhours = ${{f.hours[0]}}`);
  if (f.queues.length > 0) {{
    const ql = f.queues.map(q => `'${{q.replace(/'/g,"''")}}'`).join(',');
    parts.push(`${{a}}entry_queue IN (${{ql}})`);
  }}
  return 'WHERE ' + parts.join(' AND ');
}}

// ═══════════════════════════════════════════════════════
// QUERY HELPER
// ═══════════════════════════════════════════════════════
async function q(sql) {{
  const r = await CONN.query(sql);
  return r.toArray().map(row => {{
    const obj = {{}};
    for (const [k, v] of Object.entries(row.toJSON())) {{
      obj[k] = typeof v === 'bigint' ? Number(v) : v;
    }}
    return obj;
  }});
}}

// ═══════════════════════════════════════════════════════
// FILTER POPULATE
// ═══════════════════════════════════════════════════════
function populateQueueFilter() {{
  const sel = document.getElementById('f-queue');
  sel.innerHTML = ALL_QUEUES.map(q => `<option value="${{q}}">${{q}}</option>`).join('');
}}
function populateQIQueues() {{
  const sel = document.getElementById('qi-queue-select');
  const jSel = document.getElementById('journey-queue-select');
  const opts = ALL_QUEUES.map(q => `<option value="${{q}}">${{q}}</option>`).join('');
  sel.innerHTML = opts;
  jSel.innerHTML = opts;
}}

// ═══════════════════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════════════════
window.switchTab = function(tab, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  btn.classList.add('active');
  // Lazy-render tab on switch
  const f = getFilterState();
  if (tab === 'process') renderProcess(f);
  else if (tab === 'cost')    renderCost(f);
  else if (tab === 'hours')   renderHeatmap(f);
  else if (tab === 'queue')   renderQueueIntel();
  else if (tab === 'journey') renderJourney();
  else if (tab === 'explorer') renderExplorer();
}};

// ═══════════════════════════════════════════════════════
// APPLY FILTERS (re-renders current tab + overview)
// ═══════════════════════════════════════════════════════
window.applyFilters = async function() {{
  const f = getFilterState();
  const activeTab = document.querySelector('.tab-btn.active')?.textContent?.trim() || 'Overview';
  await renderOverview(f);
  const tabId = document.querySelector('.tab-panel.active')?.id?.replace('tab-','') || 'overview';
  if (tabId === 'process')   renderProcess(f);
  else if (tabId === 'cost')    renderCost(f);
  else if (tabId === 'hours')   renderHeatmap(f);
  else if (tabId === 'queue')   renderQueueIntel();
  else if (tabId === 'journey') renderJourney();
  else if (tabId === 'explorer') renderExplorer();
}};

// ═══════════════════════════════════════════════════════
// KPI HELPERS
// ═══════════════════════════════════════════════════════
function kpiCard(label, value, cls) {{
  return `<div class="col-6 col-md-3">
    <div class="kpi-card ${{cls}}"><h4>${{label}}</h4><h2>${{value}}</h2></div></div>`;
}}

// ═══════════════════════════════════════════════════════
// TAB 1: OVERVIEW
// ═══════════════════════════════════════════════════════
async function renderOverview(f) {{
  const w = buildWhere(f);
  const rows = await q(`
    SELECT COUNT(*) as total,
      AVG(CASE WHEN transfers=0 THEN 1.0 ELSE 0 END)*100 as drr,
      AVG(transfers) as avg_xfer,
      MEDIAN(total_active_aht) as med_aht,
      AVG(CASE WHEN transfers>=2 THEN 1.0 ELSE 0 END)*100 as multi_rate,
      AVG(loop_flag)*100 as loop_rate
    FROM cases ${{w}}`);
  const d = rows[0] || {{}};
  const n = (d.total || 0).toLocaleString();
  document.getElementById('case-count-badge').textContent = n + ' cases';
  document.getElementById('overview-kpis').innerHTML = [
    kpiCard('Total Cases', n, 'kpi-primary'),
    kpiCard('Direct Resolution Rate', (d.drr||0).toFixed(1)+'%', 'kpi-success'),
    kpiCard('Avg Transfers', (d.avg_xfer||0).toFixed(2), 'kpi-warning'),
    kpiCard('Median AHT', Math.round(d.med_aht||0)+' min', 'kpi-purple'),
    kpiCard('Multi-Transfer Rate', (d.multi_rate||0).toFixed(1)+'%', 'kpi-danger'),
    kpiCard('Loop Rate', (d.loop_rate||0).toFixed(1)+'%', 'kpi-info'),
  ].join('');

  const xfer = await q(`SELECT transfer_bin, COUNT(*) as n FROM cases ${{w}} GROUP BY transfer_bin ORDER BY transfer_bin`);
  const labels = ['0','1','2','3+'];
  const counts = labels.map(l => (xfer.find(r=>r.transfer_bin===l)||{{n:0}}).n);
  Plotly.react('chart-ov-transfers', [{{
    type:'bar', x:labels, y:counts,
    marker:{{color:['#107C10','#FFB900','#E8820C','#E81123']}},
    text:counts.map(v=>v.toLocaleString()), textposition:'outside',
  }}], {{
    title:'Cases by Transfer Count', height:320, margin:{{t:50,l:50,r:20,b:40}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    xaxis:{{showgrid:false}}, yaxis:{{showgrid:true,gridcolor:'#EDEBE9'}},
  }}, {{responsive:true}});

  const seg = await q(`SELECT segment, COUNT(*) as n FROM cases ${{w}} GROUP BY segment`);
  Plotly.react('chart-ov-segment', [{{
    type:'pie', labels:seg.map(r=>r.segment), values:seg.map(r=>r.n),
    hole:0.4, marker:{{colors:[COLORS.primary, COLORS.purple]}},
    textinfo:'label+percent',
  }}], {{
    title:'Cases by Segment', height:320, margin:{{t:50,l:20,r:20,b:20}},
    paper_bgcolor:'transparent',
  }}, {{responsive:true}});
}}

// ═══════════════════════════════════════════════════════
// TAB 2: PROCESS & ROUTING
// ═══════════════════════════════════════════════════════
async function renderProcess(f) {{
  const w = buildWhere(f, 'c');
  const stats = await q(`SELECT COUNT(*) as total, AVG(transfers) as avg_xfer,
    AVG(CASE WHEN transfers=0 THEN 1.0 ELSE 0.0 END)*100 as drr,
    MEDIAN(routing_days) as med_routing
    FROM cases c ${{w}}`);
  const d = stats[0] || {{}};
  document.getElementById('process-kpis').innerHTML = [
    kpiCard('Total Cases', (d.total||0).toLocaleString(), 'kpi-primary'),
    kpiCard('Direct Resolution Rate', (d.drr||0).toFixed(1)+'%', 'kpi-success'),
    kpiCard('Avg Transfers', (d.avg_xfer||0).toFixed(2), 'kpi-warning'),
    kpiCard('Median Routing Days', (d.med_routing||0).toFixed(1), 'kpi-info'),
  ].join('');

  // Intermediary queues Pareto
  const where_cases = buildWhere(f, 'c');
  const pareto = await q(`
    WITH max_orders AS (
      SELECT CASE_ID, MAX(QUEUE_ORDER) as max_ord FROM transitions GROUP BY CASE_ID
    )
    SELECT t.QUEUE_NEW, COUNT(*) as n
    FROM transitions t
    JOIN max_orders m ON t.CASE_ID = m.CASE_ID
    WHERE t.QUEUE_ORDER > 1 AND t.QUEUE_ORDER < m.max_ord
      AND t.CASE_ID IN (SELECT CASE_ID FROM cases c ${{where_cases}})
    GROUP BY t.QUEUE_NEW ORDER BY n DESC LIMIT 15`);
  Plotly.react('chart-pareto', [{{
    type:'bar', x:pareto.map(r=>r.n), y:pareto.map(r=>r.QUEUE_NEW),
    orientation:'h', marker:{{color:COLORS.danger}},
    text:pareto.map(r=>r.n.toLocaleString()), textposition:'outside',
  }}], {{
    title:'Top Intermediary Queues (Pareto)', height:420,
    margin:{{t:50,l:200,r:60,b:40}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    yaxis:{{autorange:'reversed',tickfont:{{size:10}}}},
    xaxis:{{showgrid:true,gridcolor:'#EDEBE9'}},
  }}, {{responsive:true}});

  const entry = await q(`SELECT entry_queue, COUNT(*) as n FROM cases c ${{where_cases}}
    GROUP BY entry_queue ORDER BY n DESC LIMIT 12`);
  Plotly.react('chart-entry-dist', [{{
    type:'bar', x:entry.map(r=>r.n), y:entry.map(r=>r.entry_queue),
    orientation:'h', marker:{{color:COLORS.primary}},
    text:entry.map(r=>r.n.toLocaleString()), textposition:'outside',
  }}], {{
    title:'Cases by Entry Queue', height:420,
    margin:{{t:50,l:200,r:60,b:40}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    yaxis:{{autorange:'reversed',tickfont:{{size:10}}}},
    xaxis:{{showgrid:true,gridcolor:'#EDEBE9'}},
  }}, {{responsive:true}});
}}

// ═══════════════════════════════════════════════════════
// TAB 3: COST & EFFORT
// ═══════════════════════════════════════════════════════
let COST_BIN_CASES = {{}};  // transfer_bin -> [case_ids]

async function renderCost(f) {{
  const w = buildWhere(f);
  const stats = await q(`
    SELECT transfer_bin,
      QUANTILE_CONT(total_active_aht, 0.25) as q1,
      MEDIAN(total_active_aht) as med,
      QUANTILE_CONT(total_active_aht, 0.75) as q3,
      QUANTILE_CONT(total_active_aht, 0.95) as p95,
      AVG(total_active_aht) as mean_val, COUNT(*) as n,
      QUANTILE_CONT(messages, 0.25) as mq1,
      MEDIAN(messages) as mmed,
      QUANTILE_CONT(messages, 0.75) as mq3,
      QUANTILE_CONT(messages, 0.95) as mp95,
      AVG(messages) as mmean
    FROM cases ${{w}} AND total_active_aht > 0
    GROUP BY transfer_bin ORDER BY transfer_bin`);

  const baseline_aht = stats.find(r=>r.transfer_bin==='0');
  const high_aht     = stats.find(r=>r.transfer_bin==='3+');
  const aht_pct = baseline_aht && high_aht && baseline_aht.med > 0
    ? ((high_aht.med / baseline_aht.med - 1) * 100) : 0;
  const baseline_msg = stats.find(r=>r.transfer_bin==='0');
  const high_msg = stats.find(r=>r.transfer_bin==='3+');
  const msg_pct = baseline_msg && high_msg && baseline_msg.mmed > 0
    ? ((high_msg.mmed / baseline_msg.mmed - 1) * 100) : 0;

  document.getElementById('cost-guide-stmt').innerHTML =
    `Every transfer doesn't just delay the customer, <strong>it inflates the total effort.</strong>
     A case that gets transferred 3+ times costs <strong>${{Math.round(aht_pct)}}% more handle time</strong>
     and generates <strong>${{Math.round(msg_pct)}}% more customer messages</strong>
     than one resolved first-touch. <strong>This is the compounding cost of mis-routing.</strong>`;

  document.getElementById('cost-kpis').innerHTML = [
    kpiCard('AHT — First Touch', Math.round(baseline_aht?.med||0)+' min', 'kpi-success'),
    kpiCard('AHT — 3+ Transfers', Math.round(high_aht?.med||0)+' min', 'kpi-danger'),
    kpiCard('Messages — First Touch', Math.round(baseline_msg?.mmed||0), 'kpi-success'),
    kpiCard('Messages — 3+ Transfers', Math.round(high_msg?.mmed||0), 'kpi-danger'),
  ].join('');

  document.getElementById('cost-insight').innerHTML =
    `Every additional transfer inflates handle time by ~${{Math.round(aht_pct/3)}}% per step
     and customer messages by ~${{Math.round(msg_pct/3)}}% per step.`;

  // Build box traces
  const bins = ['0','1','2','3+'];
  const binLabels = ['0 transfers','1 transfer','2 transfers','3+ transfers'];
  const ahtTraces = [], msgTraces = [];

  // Store bin cases for click
  const binCases = await q(`SELECT transfer_bin, CAST(CASE_ID AS VARCHAR) as cid
    FROM cases ${{w}} AND total_active_aht > 0`);
  COST_BIN_CASES = {{}};
  for (const r of binCases) {{
    if (!COST_BIN_CASES[r.transfer_bin]) COST_BIN_CASES[r.transfer_bin] = [];
    COST_BIN_CASES[r.transfer_bin].push(r.cid);
  }}

  for (const bin of bins) {{
    const row = stats.find(r=>r.transfer_bin===bin);
    if (!row) continue;
    const label = bin + (bin==='1'?' transfer':' transfers');
    const p95_aht = row.p95;
    const iqr_aht = row.q3 - row.q1;
    const color = BIN_COLORS[bin];

    ahtTraces.push({{
      type:'box', name:label,
      q1:[row.q1], median:[row.med], q3:[row.q3], mean:[row.mean_val],
      lowerfence:[Math.max(row.q1 - 1.5*iqr_aht, 0)],
      upperfence:[Math.min(row.q3 + 1.5*iqr_aht, p95_aht)],
      boxmean:true, fillcolor:color+'55', line:{{color}}, marker:{{color}},
    }});

    const iqr_msg = row.mq3 - row.mq1;
    msgTraces.push({{
      type:'box', name:label,
      q1:[row.mq1], median:[row.mmed], q3:[row.mq3], mean:[row.mmean],
      lowerfence:[Math.max(row.mq1 - 1.5*iqr_msg, 0)],
      upperfence:[Math.min(row.mq3 + 1.5*iqr_msg, row.mp95)],
      boxmean:true, fillcolor:color+'55', line:{{color}}, marker:{{color}},
    }});
  }}

  const boxLayout = (title, ytitle) => ({{
    title, yaxis_title:ytitle, height:400, showlegend:false,
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    yaxis:{{showgrid:true,gridcolor:'#EDEBE9'}},
    xaxis:{{showgrid:false}}, margin:{{t:50,l:60,r:30,b:40}},
  }});

  const ahtDiv = document.getElementById('chart-aht-box');
  Plotly.react(ahtDiv, ahtTraces, boxLayout('Handle Time by Transfer Count','AHT (min)'), {{responsive:true}});
  ahtDiv.on('plotly_click', data => showCostModal(data, 'AHT'));

  const msgDiv = document.getElementById('chart-msg-box');
  Plotly.react(msgDiv, msgTraces, boxLayout('Customer Messages by Transfer Count','Messages'), {{responsive:true}});
  msgDiv.on('plotly_click', data => showCostModal(data, 'Messages'));

  // Multiplier effect
  const esc = await q(`SELECT transfer_bin, MEDIAN(total_active_aht) as aht, MEDIAN(messages) as msg
    FROM cases ${{w}} GROUP BY transfer_bin ORDER BY transfer_bin`);
  const base_a = esc.find(r=>r.transfer_bin==='0')?.aht || 1;
  const base_m = esc.find(r=>r.transfer_bin==='0')?.msg || 1;
  Plotly.react('chart-multiplier', [
    {{type:'bar', name:'Handle Time (indexed)', x:esc.map(r=>r.transfer_bin),
      y:esc.map(r=>r.aht/base_a*100), marker:{{color:COLORS.primary}},
      text:esc.map(r=>Math.round(r.aht/base_a*100)), textposition:'outside'}},
    {{type:'bar', name:'Messages (indexed)', x:esc.map(r=>r.transfer_bin),
      y:esc.map(r=>r.msg/base_m*100), marker:{{color:COLORS.warning}},
      text:esc.map(r=>Math.round(r.msg/base_m*100)), textposition:'outside'}},
  ], {{
    title:'Multiplier Effect: Handle Time & Messages vs First-Touch Baseline',
    barmode:'group', height:380,
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    yaxis:{{showgrid:true,gridcolor:'#EDEBE9'}},
    shapes:[{{type:'line',x0:-0.5,x1:3.5,y0:100,y1:100,line:{{dash:'dash',color:'#999'}}}}],
    annotations:[{{x:3.5,y:100,text:'Baseline = 100',showarrow:false,font:{{size:10,color:'#888'}},xanchor:'left'}}],
    margin:{{t:55,l:60,r:20,b:50}}, legend:{{orientation:'h',y:1.1}},
  }}, {{responsive:true}});
}}

async function showCostModal(data, chartType) {{
  const curveIdx = data.points[0].curveNumber;
  const bins = ['0','1','2','3+'];
  const bin = bins[curveIdx];
  const cids = COST_BIN_CASES[bin] || [];
  const label = bin + (bin==='1'?' transfer':' transfers');
  await showCaseModal(`${{chartType}}: ${{cids.length.toLocaleString()}} cases — ${{label}}`, cids);
}}

// ═══════════════════════════════════════════════════════
// TAB 4: HOURS & TRANSFER HEATMAP
// ═══════════════════════════════════════════════════════
window.setHeatmapView = function(view, btn) {{
  HEATMAP_VIEW = view;
  document.querySelectorAll('.toggle-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  if (HEATMAP_DATA) drawHeatmap();
}};

async function renderHeatmap(f) {{
  const w = buildWhere(f);
  const stats = await q(`SELECT COUNT(*) as total,
    AVG(CASE WHEN inhours=0 THEN 1.0 ELSE 0.0 END)*100 as ooh_rate,
    AVG(CASE WHEN inhours=0 AND transfers>=2 THEN 1.0 ELSE 0.0 END)*100 as ooh_multi,
    AVG(CASE WHEN inhours=1 AND transfers>=2 THEN 1.0 ELSE 0.0 END)*100 as ih_multi
    FROM cases ${{w}}`);
  const d = stats[0] || {{}};

  document.getElementById('hours-guide-stmt').innerHTML =
    `Out-of-hours cases don't just transfer more often, <strong>they transfer harder.</strong>
     The OOH multi-transfer rate is <strong>${{Math.round(d.ooh_multi||0)}}% vs ${{Math.round(d.ih_multi||0)}}% in-hours.</strong>
     The heatmap below reveals exactly when the routing breaks down across the week.`;

  document.getElementById('hours-kpis').innerHTML = [
    kpiCard('In-Hours Multi-Transfer %', Math.round(d.ih_multi||0)+'%', 'kpi-success'),
    kpiCard('OOH Multi-Transfer %', Math.round(d.ooh_multi||0)+'%', 'kpi-danger'),
    kpiCard('OOH Case Rate', Math.round(d.ooh_rate||0)+'%', 'kpi-info'),
    kpiCard('Total Cases', (d.total||0).toLocaleString(), 'kpi-primary'),
  ].join('');

  // Fetch heatmap data — % of day
  const hmRows = await q(`
    WITH raw AS (
      SELECT day_of_week, hour_of_day,
        COUNT(*) as volume,
        SUM(total_active_aht) as sum_aht,
        SUM(messages) as sum_msgs,
        SUM(routing_days) as sum_routing,
        AVG(CAST(inhours AS DOUBLE)) as inhours_rate
      FROM cases ${{w}} AND day_of_week IS NOT NULL AND hour_of_day IS NOT NULL
      GROUP BY day_of_week, hour_of_day
    ),
    day_sums AS (
      SELECT day_of_week, SUM(volume) as dv, SUM(sum_aht) as da,
        SUM(sum_msgs) as dm, SUM(sum_routing) as dr
      FROM raw GROUP BY day_of_week
    )
    SELECT r.day_of_week, r.hour_of_day,
      CASE WHEN d.dv>0 THEN r.volume*100.0/d.dv ELSE 0 END as volume_pct,
      CASE WHEN d.da>0 THEN r.sum_aht*100.0/d.da ELSE 0 END as aht_pct,
      CASE WHEN d.dm>0 THEN r.sum_msgs*100.0/d.dm ELSE 0 END as msgs_pct,
      CASE WHEN d.dr>0 THEN r.sum_routing*100.0/d.dr ELSE 0 END as routing_pct,
      r.inhours_rate
    FROM raw r JOIN day_sums d ON r.day_of_week=d.day_of_week`);

  // Build 7x24 grids
  const grids = {{}};
  for (const key of ['volume_pct','aht_pct','msgs_pct','routing_pct','inhours_rate']) {{
    grids[key] = Array.from({{length:7}}, ()=>Array(24).fill(0));
  }}
  for (const row of hmRows) {{
    const di = Number(row.day_of_week), hi = Number(row.hour_of_day);
    if (di >= 0 && di < 7 && hi >= 0 && hi < 24) {{
      grids.volume_pct[di][hi] = row.volume_pct;
      grids.aht_pct[di][hi] = row.aht_pct;
      grids.msgs_pct[di][hi] = row.msgs_pct;
      grids.routing_pct[di][hi] = row.routing_pct;
      grids.inhours_rate[di][hi] = (row.inhours_rate || 0) * 100;
    }}
  }}
  HEATMAP_DATA = grids;
  drawHeatmap();
}}

function drawHeatmap() {{
  const viewMap = {{
    volume:'volume_pct', aht:'aht_pct', messages:'msgs_pct',
    routing:'routing_pct', inhours:'inhours_rate',
  }};
  const titleMap = {{
    volume:'Transfer Volume (% of day)', aht:'Handle Time (% of day)',
    messages:'Customer Messages (% of day)', routing:'Routing Wait (% of day)',
    inhours:'In-Hours Rate (%)',
  }};
  const key = viewMap[HEATMAP_VIEW];
  const vals = HEATMAP_DATA[key];

  Plotly.react('chart-heatmap', [{{
    type:'heatmap', z:vals, x:HOUR_LABELS, y:DAY_NAMES,
    colorscale:RED_SCALE, showscale:true,
    colorbar:{{title:{{text:'% of day',font:{{size:10}}}},thickness:12,len:0.85}},
    xgap:2, ygap:2,
    hovertemplate:'%{{y}}, %{{x}}<br>%{{z:.1f}}%<extra></extra>',
    text:vals.map(row=>row.map(v=>v.toFixed(1)+'%')),
    texttemplate:'%{{text}}', textfont:{{size:8}},
  }}], {{
    title:titleMap[HEATMAP_VIEW] + ' — each row normalised to 100% of that day',
    height:500, margin:{{l:100,r:60,t:55,b:50}},
    xaxis:{{title:'Hour of Day',tickfont:{{size:9}},dtick:1}},
    yaxis:{{tickfont:{{size:11}},autorange:'reversed'}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  }}, {{responsive:true}});
}}

// ═══════════════════════════════════════════════════════
// TAB 5: QUEUE INTELLIGENCE
// ═══════════════════════════════════════════════════════
window.renderQueueIntel = async function() {{
  const f = getFilterState();
  const w = buildWhere(f, 'c');
  const selQ = document.getElementById('qi-queue-select').value;
  if (!selQ) return;

  const stats = await q(`SELECT COUNT(*) as n,
    AVG(CASE WHEN transfers=0 THEN 1.0 ELSE 0.0 END)*100 as drr,
    AVG(transfers) as avg_xfer, MEDIAN(total_active_aht) as med_aht,
    MEDIAN(routing_days) as med_routing
    FROM cases c ${{w}} AND c.entry_queue='${{selQ.replace(/'/g,"''")}}'`);
  const d = stats[0] || {{}};
  document.getElementById('qi-kpis').innerHTML = [
    kpiCard('Cases Through Queue', (d.n||0).toLocaleString(), 'kpi-primary'),
    kpiCard('Direct Resolution Rate', (d.drr||0).toFixed(1)+'%', 'kpi-success'),
    kpiCard('Avg Transfers', (d.avg_xfer||0).toFixed(2), 'kpi-warning'),
    kpiCard('Median AHT', Math.round(d.med_aht||0)+' min', 'kpi-info'),
  ].join('');

  const wCases = buildWhere(f, 'c');
  const inbound = await q(`
    SELECT prev.QUEUE_NEW as from_queue, COUNT(*) as n
    FROM transitions t
    JOIN transitions prev ON t.CASE_ID=prev.CASE_ID AND t.QUEUE_ORDER=prev.QUEUE_ORDER+1
    WHERE t.QUEUE_NEW='${{selQ.replace(/'/g,"''")}}'
      AND t.CASE_ID IN (SELECT CASE_ID FROM cases c ${{wCases}})
    GROUP BY prev.QUEUE_NEW ORDER BY n DESC LIMIT 12`);

  const outbound = await q(`
    SELECT nxt.QUEUE_NEW as to_queue, COUNT(*) as n
    FROM transitions t
    JOIN transitions nxt ON t.CASE_ID=nxt.CASE_ID AND nxt.QUEUE_ORDER=t.QUEUE_ORDER+1
    WHERE t.QUEUE_NEW='${{selQ.replace(/'/g,"''")}}'
      AND t.CASE_ID IN (SELECT CASE_ID FROM cases c ${{wCases}})
    GROUP BY nxt.QUEUE_NEW ORDER BY n DESC LIMIT 12`);

  Plotly.react('chart-qi-inbound', [{{
    type:'bar', x:inbound.map(r=>r.n), y:inbound.map(r=>r.from_queue),
    orientation:'h', marker:{{color:COLORS.teal}},
    text:inbound.map(r=>r.n.toLocaleString()), textposition:'outside',
  }}], {{
    title:`Where cases come from before ${{selQ}}`, height:380,
    margin:{{t:50,l:200,r:60,b:40}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    yaxis:{{autorange:'reversed',tickfont:{{size:9}}}},
    xaxis:{{showgrid:true,gridcolor:'#EDEBE9'}},
  }}, {{responsive:true}});

  Plotly.react('chart-qi-outbound', [{{
    type:'bar', x:outbound.map(r=>r.n), y:outbound.map(r=>r.to_queue),
    orientation:'h', marker:{{color:COLORS.purple}},
    text:outbound.map(r=>r.n.toLocaleString()), textposition:'outside',
  }}], {{
    title:`Where cases go after ${{selQ}}`, height:380,
    margin:{{t:50,l:200,r:60,b:40}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    yaxis:{{autorange:'reversed',tickfont:{{size:9}}}},
    xaxis:{{showgrid:true,gridcolor:'#EDEBE9'}},
  }}, {{responsive:true}});
}};

// ═══════════════════════════════════════════════════════
// TAB 6: JOURNEY PATHWAYS
// ═══════════════════════════════════════════════════════
window.renderJourney = async function() {{
  const f = getFilterState();
  const w = buildWhere(f, 'c');
  const selQ = document.getElementById('journey-queue-select').value;
  const depth = parseInt(document.getElementById('journey-depth').value) || 3;
  if (!selQ) return;

  const qSafe = selQ.replace(/'/g, "''");

  // Get all case journeys through this queue
  const journeyRows = await q(`
    SELECT t.CASE_ID, t.QUEUE_NEW, t.QUEUE_ORDER
    FROM transitions t
    WHERE t.CASE_ID IN (
      SELECT CASE_ID FROM transitions WHERE QUEUE_NEW='${{qSafe}}'
    ) AND t.CASE_ID IN (SELECT CASE_ID FROM cases c ${{w}})
    ORDER BY t.CASE_ID, t.QUEUE_ORDER`);

  // Group into case journeys
  const byCase = {{}};
  for (const row of journeyRows) {{
    const cid = String(row.CASE_ID);
    if (!byCase[cid]) byCase[cid] = [];
    byCase[cid].push({{queue:row.QUEUE_NEW, order:Number(row.QUEUE_ORDER)}});
  }}
  for (const cid of Object.keys(byCase)) {{
    byCase[cid].sort((a,b)=>a.order-b.order);
  }}

  const caseIds = Object.keys(byCase);
  const totalThrough = caseIds.length;

  // Build forward + backward paths
  const fwdPaths = [], fwdCids = [], bwdPaths = [], bwdCids = [];
  const completePaths = [], pathToCids = {{}};

  for (const [cid, steps] of Object.entries(byCase)) {{
    const queues = steps.map(s=>s.queue);
    const fullPath = queues.join(' → ');
    if (!pathToCids[fullPath]) pathToCids[fullPath] = [];
    pathToCids[fullPath].push(cid);
    completePaths.push(fullPath);

    const idx = queues.indexOf(selQ);
    if (idx >= 0) {{
      const fwd = queues.slice(idx, idx + depth);
      if (fwd.length > 1) {{ fwdPaths.push(fwd); fwdCids.push(cid); }}
      const bwd = queues.slice(Math.max(0, idx - depth + 1), idx + 1);
      if (bwd.length > 1) {{ bwdPaths.push(bwd); bwdCids.push(cid); }}
    }}
  }}

  // Stats
  const avgLen = completePaths.length > 0
    ? completePaths.reduce((s,p)=>s+p.split('→').length,0)/completePaths.length : 0;
  document.getElementById('journey-kpis').innerHTML = [
    kpiCard('Cases Through Queue', totalThrough.toLocaleString(), 'kpi-primary'),
    kpiCard('Unique Forward Paths', new Set(fwdPaths.map(p=>p.join('→'))).size, 'kpi-success'),
    kpiCard('Unique Backward Paths', new Set(bwdPaths.map(p=>p.join('→'))).size, 'kpi-warning'),
    kpiCard('Avg Journey Length', avgLen.toFixed(1), 'kpi-info'),
  ].join('');

  // Avoidable transfers
  let avoidable = 0;
  for (const [path, cids] of Object.entries(pathToCids)) {{
    const qs = path.split(' → ');
    if (qs.length > 1 && qs[0].trim() === qs[qs.length-1].trim()) avoidable += cids.length;
  }}
  const avDiv = document.getElementById('journey-avoidable');
  if (avoidable > 0) {{
    const pct = (avoidable/totalThrough*100).toFixed(1);
    avDiv.innerHTML = `<strong style="color:#E81123">${{avoidable.toLocaleString()}} cases (${{pct}}%) took a round trip</strong>
      back to the same queue they started in. Every one of these transfers was avoidable.`;
    avDiv.style.display = 'block';
  }} else {{
    avDiv.style.display = 'none';
  }}

  // Sankey charts
  drawSankey('chart-sankey-fwd', fwdPaths, fwdCids, `Forward Journey from ${{selQ}}`);
  drawSankey('chart-sankey-bwd', bwdPaths, bwdCids, `Backward Journey to ${{selQ}}`);

  // Top 10 paths table
  const pathCounts = Object.entries(pathToCids)
    .map(([p,ids])=>{{
      const qs = p.split(' → ');
      return {{path:p, n:ids.length, pct:(ids.length/totalThrough*100).toFixed(1), cids:ids, hops:qs.length}};
    }})
    .sort((a,b)=>b.n-a.n).slice(0,10);

  // Get cost metrics per path from DuckDB
  const allPathCids = pathCounts.flatMap(p=>p.cids).map(c=>`'${{c}}'`).join(',');
  let costMap = {{}};
  if (allPathCids) {{
    const costRows = await q(`
      SELECT CAST(CASE_ID AS VARCHAR) as cid, total_active_aht, routing_days, messages,
        CAST(transfer_bin AS VARCHAR) as transfer_bin
      FROM cases WHERE CASE_ID IN (${{allPathCids}})`);
    for (const r of costRows) {{
      costMap[r.cid] = r;
    }}
  }}

  let tableHtml = `<table class="table data-table table-sm">
    <thead><tr>
      <th>Journey Path</th><th>Cases</th><th>% Share</th>
      <th>Med AHT (min)</th><th>Med Routing (days)</th><th>Med Messages</th>
    </tr></thead><tbody>`;
  for (const row of pathCounts) {{
    const pathData = row.cids.map(c=>costMap[c]).filter(Boolean);
    const medAHT = pathData.length ? median(pathData.map(r=>r.total_active_aht||0)) : 0;
    const medRt  = pathData.length ? median(pathData.map(r=>r.routing_days||0)) : 0;
    const medMsg = pathData.length ? median(pathData.map(r=>r.messages||0)) : 0;
    const cidsJson = JSON.stringify(row.cids).replace(/"/g,'&quot;');
    tableHtml += `<tr onclick="showCaseModal('${{row.hops}}-Queue Path: ${{row.n}} cases',JSON.parse(this.dataset.cids))" data-cids="${{cidsJson}}" style="cursor:pointer">
      <td style="font-size:.75rem;max-width:400px;word-break:break-word">${{row.path}}</td>
      <td style="text-align:center">${{row.n.toLocaleString()}}</td>
      <td style="text-align:center">${{row.pct}}%</td>
      <td style="text-align:center">${{Math.round(medAHT)}}</td>
      <td style="text-align:center">${{medRt.toFixed(1)}}</td>
      <td style="text-align:center">${{Math.round(medMsg)}}</td>
    </tr>`;
  }}
  tableHtml += '</tbody></table>';
  document.getElementById('journey-path-table').innerHTML = tableHtml;
}};

function median(arr) {{
  if (!arr.length) return 0;
  const s = [...arr].sort((a,b)=>a-b);
  const m = Math.floor(s.length/2);
  return s.length%2 ? s[m] : (s[m-1]+s[m])/2;
}}

function drawSankey(divId, paths, cids, title) {{
  if (!paths.length) {{
    Plotly.react(divId,[],{{title,height:300}},{{responsive:true}});
    return;
  }}
  const linkMap = {{}};
  for (let pi=0; pi<paths.length; pi++) {{
    const path = paths[pi];
    const cid = cids[pi];
    for (let i=0; i<path.length-1; i++) {{
      const s = `${{path[i]}} (Step ${{i+1}})`;
      const t = `${{path[i+1]}} (Step ${{i+2}})`;
      const key = s+'||'+t;
      if (!linkMap[key]) linkMap[key] = {{s,t,cids:[]}};
      linkMap[key].cids.push(cid);
    }}
  }}
  const links = Object.values(linkMap);
  const allNodes = [...new Set(links.flatMap(l=>[l.s,l.t]))];
  const nodeIdx = Object.fromEntries(allNodes.map((n,i)=>[n,i]));

  // Store link→case IDs by position index (not as customdata, which breaks Plotly.js Sankey)
  const linkCidsByIdx = links.map(l=>l.cids);

  const trace = {{
    type:'sankey',
    arrangement:'snap',
    node:{{pad:15,thickness:20,label:allNodes,
      color:allNodes.map((_,i)=>CHART_COLORS[i%CHART_COLORS.length])}},
    link:{{
      source:links.map(l=>nodeIdx[l.s]),
      target:links.map(l=>nodeIdx[l.t]),
      value:links.map(l=>l.cids.length),
      label:links.map(l=>l.cids.length+' cases'),
    }},
  }};

  const div = document.getElementById(divId);
  Plotly.react(div, [trace], {{
    title:title+' (click any link to see cases)', height:480,
    margin:{{l:20,r:20,t:55,b:20}}, font:{{size:11,family:'Segoe UI'}},
    paper_bgcolor:'transparent',
  }}, {{responsive:true}});

  div.on('plotly_click', function(data) {{
    const pt = data.points[0];
    if (pt && typeof pt.pointNumber !== 'undefined') {{
      const clickedCids = linkCidsByIdx[pt.pointNumber] || [];
      if (!clickedCids.length) return;
      const srcLabel = allNodes[pt.source] ?? '';
      const tgtLabel = allNodes[pt.target] ?? '';
      showCaseModal(`${{clickedCids.length}} cases: ${{srcLabel}} \u2192 ${{tgtLabel}}`, clickedCids);
    }}
  }});
}}

// ═══════════════════════════════════════════════════════
// TAB 7: DATA EXPLORER
// ═══════════════════════════════════════════════════════
window.renderExplorer = async function() {{
  const f = getFilterState();
  let w = buildWhere(f);
  const xfer = document.getElementById('ex-xfer').value;
  if (xfer !== 'all') w += ` AND transfer_bin='${{xfer}}'`;

  const countRow = await q(`SELECT COUNT(*) as n FROM cases ${{w}}`);
  const total = countRow[0]?.n || 0;
  document.getElementById('ex-count').textContent = total.toLocaleString() + ' cases';

  const rows = await q(`
    SELECT CAST(CASE_ID AS VARCHAR) as case_id, entry_queue, final_queue,
      CAST(transfers AS INT) as transfers,
      CAST(transfer_bin AS VARCHAR) as transfer_bin,
      ROUND(total_active_aht, 0) as aht_min,
      ROUND(routing_days, 1) as routing_days,
      CAST(messages AS INT) as messages,
      segment, inhours
    FROM cases ${{w}} ORDER BY transfers DESC, total_active_aht DESC LIMIT 5000`);

  EXPLORER_DATA = rows;
  EXPLORER_PAGE = 0;
  renderExplorerPage();
}};

function renderExplorerPage() {{
  const start = EXPLORER_PAGE * PAGE_SIZE;
  const page  = EXPLORER_DATA.slice(start, start + PAGE_SIZE);
  const cols  = ['case_id','entry_queue','final_queue','transfers','aht_min',
                 'routing_days','messages','segment'];
  const headers = ['Case ID','Entry Queue','Final Queue','Transfers','AHT (min)',
                   'Routing Days','Messages','Segment'];

  let html = `<div style="overflow-x:auto"><table class="table data-table table-sm">
    <thead><tr>${{headers.map(h=>`<th>${{h}}</th>`).join('')}}</tr></thead><tbody>`;
  for (const row of page) {{
    html += `<tr onclick="showCaseModal('Case ${{row.case_id}}',${{JSON.stringify([String(row.case_id)])}})">
      ${{cols.map(c=>`<td>${{row[c]??''}}</td>`).join('')}}</tr>`;
  }}
  html += '</tbody></table></div>';
  document.getElementById('explorer-table-container').innerHTML = html;

  // Pager
  const totalPages = Math.ceil(EXPLORER_DATA.length / PAGE_SIZE);
  let pagerHtml = `<button onclick="explorerPage(${{EXPLORER_PAGE-1}})" ${{EXPLORER_PAGE===0?'disabled':''}}>‹</button>`;
  const start_p = Math.max(0, EXPLORER_PAGE-2), end_p = Math.min(totalPages, start_p+5);
  for (let i=start_p; i<end_p; i++) {{
    pagerHtml += `<button class="${{i===EXPLORER_PAGE?'active':''}}" onclick="explorerPage(${{i}})">${{i+1}}</button>`;
  }}
  pagerHtml += `<button onclick="explorerPage(${{EXPLORER_PAGE+1}})" ${{EXPLORER_PAGE>=totalPages-1?'disabled':''}}>›</button>`;
  pagerHtml += ` <span style="font-size:.75rem;color:#888">Showing ${{start+1}}-${{Math.min(start+PAGE_SIZE,EXPLORER_DATA.length)}} of ${{EXPLORER_DATA.length.toLocaleString()}} loaded</span>`;
  document.getElementById('explorer-pager').innerHTML = pagerHtml;
}}

window.explorerPage = function(p) {{
  const total = Math.ceil(EXPLORER_DATA.length / PAGE_SIZE);
  if (p < 0 || p >= total) return;
  EXPLORER_PAGE = p;
  renderExplorerPage();
}};

window.downloadCSV = function() {{
  if (!EXPLORER_DATA.length) return;
  const cols = Object.keys(EXPLORER_DATA[0]);
  const csv = [cols.join(','), ...EXPLORER_DATA.map(r=>cols.map(c=>JSON.stringify(r[c]??'')).join(','))].join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {{type:'text/csv'}}));
  a.download = 'messenger_cases.csv'; a.click();
}};

// ═══════════════════════════════════════════════════════
// CASE DETAIL MODAL
// ═══════════════════════════════════════════════════════
window.showCaseModal = async function(title, cids) {{
  document.getElementById('caseModalTitle').textContent = title;
  document.getElementById('caseModalBody').innerHTML = '<p class="text-muted">Loading...</p>';
  const modal = new bootstrap.Modal(document.getElementById('caseModal'));
  modal.show();

  const idList = cids.slice(0,500).map(c=>`'${{String(c).replace(/'/g,"''")}}'`).join(',');
  if (!idList) {{ document.getElementById('caseModalBody').innerHTML = '<p>No cases.</p>'; return; }}

  const rows = await q(`
    SELECT CAST(CASE_ID AS VARCHAR) as case_id, entry_queue, final_queue,
      CAST(transfers AS INT) as transfers,
      ROUND(total_active_aht,0) as aht_min, ROUND(routing_days,1) as routing_days,
      CAST(messages AS INT) as messages, segment
    FROM cases WHERE CAST(CASE_ID AS VARCHAR) IN (${{idList}})
    ORDER BY transfers DESC, total_active_aht DESC`);

  const n = rows.length;
  const medAHT = median(rows.map(r=>r.aht_min||0));
  const medRt  = median(rows.map(r=>r.routing_days||0));
  const medMsg = median(rows.map(r=>r.messages||0));

  const summaryHtml = `<div class="row g-2 mb-3">
    ${{kpiCard('Cases', n.toLocaleString(), 'kpi-primary')}}
    ${{kpiCard('Median AHT', Math.round(medAHT)+' min', 'kpi-danger')}}
    ${{kpiCard('Median Routing', medRt.toFixed(1)+' days', 'kpi-warning')}}
    ${{kpiCard('Median Messages', Math.round(medMsg), 'kpi-info')}}
  </div>`;

  const cols = ['case_id','entry_queue','final_queue','transfers','aht_min','routing_days','messages','segment'];
  const headers = ['Case ID','Entry Queue','Final Queue','Transfers','AHT (min)','Routing Days','Messages','Segment'];
  let tableHtml = `<div style="overflow-x:auto;max-height:400px;overflow-y:auto">
    <table class="table data-table table-sm">
    <thead><tr>${{headers.map(h=>`<th>${{h}}</th>`).join('')}}</tr></thead><tbody>`;
  for (const row of rows) {{
    tableHtml += `<tr>${{cols.map(c=>`<td>${{row[c]??''}}</td>`).join('')}}</tr>`;
  }}
  if (cids.length > 500) tableHtml += `<tr><td colspan="${{cols.length}}" class="text-muted text-center">Showing first 500 of ${{cids.length.toLocaleString()}} cases</td></tr>`;
  tableHtml += '</tbody></table></div>';
  document.getElementById('caseModalBody').innerHTML = summaryHtml + tableHtml;
}};

// ═══════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════
initDB().catch(err => {{
  document.getElementById('loading-status').textContent = 'Error: ' + err.message;
  console.error(err);
}});
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate static HTML dashboard")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--out", default="dist", help="Output directory (default: dist)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found"); sys.exit(1)

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print(f"\nLoading {data_path} ...")
    df_raw = pd.read_csv(data_path, low_memory=False)
    # Normalise column names to uppercase
    df_raw.columns = [c.upper() for c in df_raw.columns]
    # Parse dates
    for col in ["CREATED_AT", "CLOSE_DATETIME"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_datetime(df_raw[col], errors="coerce")
    print(f"  {len(df_raw):,} rows loaded")

    print("\nPreparing case-level data ...")
    df_raw, case_df = prepare_data(df_raw)
    print(f"  {len(case_df):,} unique cases")

    print("\nExporting Parquet files ...")
    export_cases(case_df, out / "cases.parquet")
    export_transitions(df_raw, out / "transitions.parquet")

    min_date = case_df["created_at"].min().strftime("%Y-%m-%d")
    max_date = case_df["created_at"].max().strftime("%Y-%m-%d")
    all_queues = sorted(case_df["entry_queue"].dropna().unique().tolist())

    print("\nGenerating index.html ...")
    html = generate_html(case_df, min_date, max_date, all_queues)
    (out / "index.html").write_text(html, encoding="utf-8")
    mb = (out / "index.html").stat().st_size / 1024 / 1024
    print(f"  index.html         {mb:.1f} MB")

    print(f"""
╔══════════════════════════════════════════════════════╗
║  Done! Three files written to: {str(out):<22}║
╠══════════════════════════════════════════════════════╣
║  index.html           dashboard app                  ║
║  cases.parquet        case-level data                ║
║  transitions.parquet  queue transition data          ║
╠══════════════════════════════════════════════════════╣
║  LOCAL TEST:                                         ║
║    cd {str(out):<12} && python3 -m http.server 8080 ║
║    Open: http://localhost:8080                       ║
╠══════════════════════════════════════════════════════╣
║  AZURE: deploy all three files to your container     ║
║  (static hosting, no server required)                ║
╚══════════════════════════════════════════════════════╝
Date range: {min_date} → {max_date}
Queues: {len(all_queues)}
""")


if __name__ == "__main__":
    main()
