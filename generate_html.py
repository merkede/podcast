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
        case["transfers"].fillna(0), bins=[-0.1, 0, 1, 2, float("inf")], labels=["0", "1", "2", "3+"]
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
    case["hour_of_day"] = pd.to_datetime(case["created_at"], errors="coerce").dt.hour
    case["day_of_week"] = case["day_of_week"].fillna(0).astype(int)
    case["hour_of_day"] = case["hour_of_day"].fillna(12).astype(int)

    return df, case


# ══════════════════════════════════════════════════════════════════════════════
# PARQUET EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_cases(case_df, path):
    df = case_df.copy()
    # Strip .0 suffix from CASE_ID (occurs when stored as float64)
    df["CASE_ID"] = df["CASE_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
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
    df["CASE_ID"] = df["CASE_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")
    mb = path.stat().st_size / 1024 / 1024
    print(f"  transitions.parquet{len(df):>10,} rows  {mb:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_html(case_df, min_date, max_date, all_queues, all_trans_queues):
    # Pre-compute overview KPIs from full data for initial display
    total = len(case_df)
    drr = (case_df["transfers"] == 0).mean() * 100
    avg_xfer = case_df["transfers"].mean()
    med_aht = case_df["total_active_aht"].median()
    multi_rate = (case_df["transfers"] >= 2).mean() * 100
    loop_rate = case_df["loop_flag"].mean() * 100

    # Escape for JSON embedding
    queues_json = json.dumps(all_queues)
    trans_queues_json = json.dumps(all_trans_queues)

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
  position:sticky;top:0;z-index:1000;}}
.hd-logo{{background:var(--hd-green);color:#fff;font-weight:800;font-size:1.1rem;padding:.3rem .8rem;
  border-radius:4px;letter-spacing:.5px;}}
.hd-title{{color:#fff;font-size:1.3rem;font-weight:600;opacity:.9;}}
/* ── Filters ── */
.filter-panel{{background:#1a1a2e;padding:1rem 1.4rem;
  border-top:2px solid var(--pbi-blue);border-bottom:2px solid var(--pbi-blue);}}
.filter-label{{font-size:.65rem;font-weight:700;color:rgba(255,255,255,.55);text-transform:uppercase;
  letter-spacing:.5px;margin-bottom:.25rem;}}
.filter-panel select,.filter-panel input[type=date]{{font-size:.82rem;
  background:rgba(255,255,255,.1);color:rgba(255,255,255,.9);
  border:1px solid rgba(255,255,255,.22);border-radius:4px;padding:.3rem .5rem;width:100%;}}
.filter-panel select option{{background:#1a1a2e;color:#fff;}}
.filter-panel select[multiple]{{height:80px;}}
.filter-panel .toggle-btn{{background:rgba(255,255,255,.1);color:rgba(255,255,255,.8);
  border:1px solid rgba(255,255,255,.25);}}
.filter-panel .toggle-btn:hover{{background:rgba(255,255,255,.2);color:#fff;}}
.filter-panel .toggle-btn.active{{background:var(--pbi-blue);color:#fff;border-color:var(--pbi-blue);}}
#f-slider-label{{color:rgba(255,255,255,.5)!important;}}
input[type=range]{{-webkit-appearance:none;appearance:none;height:4px;border-radius:2px;
  background:rgba(255,255,255,.25);outline:none;}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--pbi-blue);cursor:pointer;}}
/* ── Tabs ── */
.tab-nav{{background:#1a1a2e;padding:.55rem 1rem;
  position:sticky;top:53px;z-index:999;display:flex;gap:.45rem;overflow-x:auto;
  align-items:center;box-shadow:0 2px 6px rgba(0,0,0,.3);}}
.tab-btn{{border:2px solid rgba(255,255,255,.18);background:rgba(255,255,255,.07);
  color:rgba(255,255,255,.75);padding:.42rem 0;font-size:.78rem;font-weight:700;
  letter-spacing:.3px;cursor:pointer;border-radius:6px;white-space:nowrap;
  transition:all .18s;flex:1;min-width:0;text-align:center;}}
.tab-btn:hover:not(.active){{background:rgba(255,255,255,.15);color:#fff;
  border-color:rgba(255,255,255,.4);}}
.tab-btn.active{{background:var(--pbi-blue);color:#fff;border-color:var(--pbi-blue);
  box-shadow:0 2px 8px rgba(0,120,212,.5);}}
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
.guide-stmt{{background:linear-gradient(135deg,#EBF3FB 0%,#DDEEFF 100%);
  border-left:5px solid var(--pbi-blue);border-radius:0 8px 8px 0;
  padding:1rem 1.4rem;margin-bottom:1.4rem;font-size:.9rem;color:#1a3a5c;
  line-height:1.65;box-shadow:0 2px 8px rgba(0,120,212,.1);}}
.guide-stmt::before{{content:'ℹ ';font-weight:700;color:var(--pbi-blue);font-style:normal;}}
/* ── Chart insight ── */
.chart-insight{{background:#FFFBF0;border-left:4px solid var(--pbi-warning);border-radius:0 6px 6px 0;
  padding:.6rem 1rem;margin-top:.6rem;font-size:.78rem;color:#5a4a00;line-height:1.5;}}
.chart-insight::before{{content:'What this shows';display:block;font-weight:700;
  color:var(--pbi-blue);font-size:.72rem;letter-spacing:.4px;text-transform:uppercase;
  margin-bottom:.3rem;}}
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
/* ── Definitions tab ── */
.def-section-title{{font-size:.68rem;font-weight:800;text-transform:uppercase;
  letter-spacing:.6px;color:#605E5C;margin:1.4rem 0 .6rem;padding-bottom:.3rem;
  border-bottom:2px solid var(--pbi-blue);}}
.def-card{{background:#fff;border-radius:8px;padding:.9rem 1.1rem;
  box-shadow:0 1.6px 3.6px rgba(0,0,0,.12);margin-bottom:.8rem;
  border-left:4px solid transparent;}}
.def-card.primary{{border-left-color:var(--pbi-blue);}}
.def-card.success{{border-left-color:var(--pbi-success);}}
.def-card.danger{{border-left-color:var(--pbi-danger);}}
.def-card.warning{{border-left-color:#B8860B;}}
.def-card.info{{border-left-color:var(--pbi-purple);}}
.def-card.teal{{border-left-color:var(--pbi-teal);}}
.def-card .def-term{{font-size:.8rem;font-weight:700;color:#201F1E;margin-bottom:.25rem;}}
.def-card .def-formula{{background:#F3F2F1;border-radius:4px;padding:.25rem .55rem;
  font-family:'Courier New',monospace;font-size:.73rem;color:#444;margin:.4rem 0;display:inline-block;}}
.def-card .def-body{{font-size:.8rem;color:#444;line-height:1.6;}}
.def-card .def-example{{font-size:.75rem;color:#107C10;font-style:italic;margin-top:.3rem;}}
/* ── Fixed 4Cs callout ── */
#fcs-callout{{position:fixed;bottom:1.5rem;right:1.5rem;z-index:1050;
  background:linear-gradient(135deg,#fff8e1,#fff3cd);border:1.5px solid #FFB900;
  border-radius:8px;padding:.55rem 1rem;font-size:.78rem;color:#5a4000;
  display:flex;align-items:center;gap:.6rem;
  box-shadow:0 3px 12px rgba(255,185,0,.4);white-space:nowrap;}}
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
  <img src="hastings_logo.svg" alt="Hastings Direct" height="44"
       style="object-fit:contain;display:block;">
  <div>
    <div class="hd-title">Messenger Transfer Analytics &nbsp;|&nbsp; Hastings Direct</div>
    <div class="hd-title" style="opacity:.7;margin-top:.15rem;font-size:.95rem;">by Hamzah Javaid</div>
  </div>
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
      <div class="d-flex gap-1 mt-1" style="align-items:center;">
        <input type="range" id="f-start-slider" min="0" max="0" value="0"
          oninput="updateDateFromSlider('start',this.value)" style="flex:1;height:4px;">
        <input type="range" id="f-end-slider" min="0" max="0" value="0"
          oninput="updateDateFromSlider('end',this.value)" style="flex:1;height:4px;">
      </div>
      <div id="f-slider-label" style="font-size:.62rem;color:#888;margin-top:.2rem;text-align:center;"></div>
    </div>
    <div class="col-md-3">
      <div class="filter-label">Entry Queue</div>
      <select id="f-queue" multiple title="All queues (select to filter)"></select>
    </div>
    <div class="col-md-2">
      <div class="filter-label">Hours</div>
      <div class="d-flex gap-1 mb-1">
        <button onclick="setHours('1')" class="toggle-btn" style="flex:1;font-size:.72rem;">In-Hours</button>
        <button onclick="setHours('0')" class="toggle-btn" style="flex:1;font-size:.72rem;">Out-of-Hours</button>
      </div>
      <select id="f-hours" multiple>
        <option value="1" selected>In-Hours</option>
        <option value="0" selected>Out-of-Hours</option>
      </select>
    </div>
    <div class="col-md-2">
      <div class="filter-label">Segment</div>
      <div class="d-flex gap-1 mb-1">
        <button onclick="setSegment('Retail')" class="toggle-btn" style="flex:1;font-size:.72rem;">Retail</button>
        <button onclick="setSegment('Claims')" class="toggle-btn" style="flex:1;font-size:.72rem;">Claims</button>
      </div>
      <select id="f-segment" multiple>
        <option value="Retail" selected>Retail</option>
        <option value="Claims" selected>Claims</option>
      </select>
    </div>
    <div class="col-md-2 d-flex align-items-end gap-1 flex-wrap">
      <button onclick="applyFilters()" class="btn btn-sm btn-primary w-100">Apply Filters</button>
      <button onclick="resetFilters()" class="btn btn-sm btn-outline-light w-100">Reset Filters</button>
    </div>
  </div>
</div>

<!-- TABS -->
<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
  <button class="tab-btn" onclick="switchTab('process',this)">Process &amp; Routing</button>
  <button class="tab-btn" onclick="switchTab('cost',this)">Cost &amp; Effort</button>
  <button class="tab-btn" onclick="switchTab('hours',this)">Heatmaps</button>
  <button class="tab-btn" onclick="switchTab('queue',this)">Queue Intelligence</button>
  <button class="tab-btn" onclick="switchTab('explorer',this)">Data Explorer</button>
  <button class="tab-btn" onclick="switchTab('definitions',this)">Definitions</button>
</div>

<!-- TAB CONTENT -->
<div class="content-area">

  <!-- ── TAB 1: OVERVIEW ── -->
  <div id="tab-overview" class="tab-panel active">
    <div class="row g-3" id="overview-kpis"></div>

    <!-- ── Journey Pathways (embedded in Overview) ── -->
    <hr style="margin:1.5rem 0;border-color:#C8C6C4;">
    <h5 style="font-weight:800;color:#201F1E;font-size:.95rem;margin-bottom:.8rem;">Journey Pathways</h5>
    <div class="guide-stmt" style="margin-bottom:1rem;">
      <strong>The shortest path to resolution is the cheapest one.</strong>
      This section maps how Messenger cases actually flow through the business.
      Every extra hop on the journey is time, effort, and customer patience burned.
    </div>
    <div class="row g-2 mb-3">
      <div class="col-md-4">
        <div class="filter-label" style="color:#605E5C;">Select a Queue</div>
        <select id="journey-queue-select" class="form-select form-select-sm" onchange="renderJourney()"></select>
      </div>
      <div class="col-md-2">
        <div class="filter-label" style="color:#605E5C;">Number of Transfers</div>
        <select id="journey-depth" class="form-select form-select-sm" onchange="renderJourney()">
          <option value="2">2</option><option value="3" selected>3</option>
          <option value="4">4</option><option value="5">5</option>
        </select>
      </div>
    </div>
    <div class="row g-3" id="journey-kpis"></div>
    <div id="journey-avoidable" class="insight-card" style="display:none;"></div>
    <div class="chart-card mt-2">
      <div id="chart-sankey-fwd"></div>
      <div class="chart-insight">Band width = number of cases flowing along that path. Thicker bands are the dominant routing patterns. Click any band to see the individual cases behind that flow. Queues that appear as large hubs with many outbound paths are routing bottlenecks.</div>
    </div>
    <div class="chart-card mt-2">
      <div id="chart-sankey-bwd"></div>
      <div class="chart-insight">The reverse view — tracing backwards from the selected queue. Shows which upstream routing decisions led to cases arriving here. Repeated upstream queues suggest a systematic mis-route that could be corrected at source.</div>
    </div>
    <div class="mt-2">
      <h6 style="font-weight:700;">Top 10 Complete Paths <small class="text-muted">(click row for case detail)</small></h6>
      <div id="journey-path-table"></div>
    </div>
  </div>

  <!-- ── TAB 2: PROCESS & ROUTING ── -->
  <div id="tab-process" class="tab-panel">
    <div class="guide-stmt">
      <strong>Not all queues add value, some just add delay.</strong>
      The intermediary queues shown here are where Messenger cases sit waiting between handoffs,
      contributing nothing to resolution. If a queue appears frequently in the Pareto Distribution,
      it's either a structural bottleneck or a sign that cases are being sent there by mistake.
    </div>
    <div class="row g-3" id="process-kpis"></div>
    <div class="row g-3 mt-1">
      <div class="col-md-6"><div class="chart-card">
        <div id="chart-pareto"></div>
        <div class="chart-insight">Ranks queues by total days of delay they accumulate across all cases — not just how often they appear. The cumulative % line shows which queues together account for 80% of all routing delay. Fixing the top 2–3 delivers the most impact.</div>
      </div></div>
      <div class="col-md-6"><div class="chart-card">
        <div id="chart-entry-dist"></div>
        <div class="chart-insight">For each starting queue, shows what % of its cases were resolved first-touch (green) vs transferred at least once (red). Queues with high red bars are mis-routing most of their cases to the wrong team from the outset.</div>
      </div></div>
    </div>
  </div>

  <!-- ── TAB 3: COST & EFFORT ── -->
  <div id="tab-cost" class="tab-panel">
    <div class="guide-stmt" id="cost-guide-stmt">Loading...</div>
    <div class="row g-3" id="cost-kpis"></div>
    <div id="cost-insight" class="insight-card"></div>
    <div class="row g-3">
      <div class="col-md-6"><div class="chart-card">
        <div id="chart-aht-box"></div>
        <div class="chart-insight">Box = middle 50% of cases (IQR). Line = median. Diamond = mean. Whiskers = typical range. Click any box to drill into the individual cases in that group. If the 3+ box sits much higher than the 0 box, transfers are directly inflating handle time.</div>
      </div></div>
      <div class="col-md-6"><div class="chart-card">
        <div id="chart-msg-box"></div>
        <div class="chart-insight">Same spread chart for customer message volume. More transfers means more back-and-forth messages — each handoff resets customer expectations and generates additional contact. Click any box to see those cases.</div>
      </div></div>
    </div>
  </div>

  <!-- ── TAB 4: HEATMAPS ── -->
  <div id="tab-hours" class="tab-panel">
    <div class="toggle-group" id="heatmap-toggles">
      <button class="toggle-btn active" onclick="setHeatmapView('volume',this)">Transfer Volume</button>
      <button class="toggle-btn" onclick="setHeatmapView('aht',this)">Median AHT</button>
      <button class="toggle-btn" onclick="setHeatmapView('messages',this)">Customer Messages</button>
      <button class="toggle-btn" onclick="setHeatmapView('routing',this)">Routing Wait</button>
      <button class="toggle-btn" onclick="setHeatmapView('inhours',this)">In/Out Hours</button>
    </div>
    <div class="chart-card">
      <div id="chart-heatmap"></div>
      <div class="chart-insight">Each cell shows the selected metric as a % of that day's total — so each row always sums to 100%. Darker red = a higher concentration of volume, AHT, messages, or routing delay at that hour relative to the rest of the day. Use this to identify when routing breaks down, not just how much.</div>
    </div>
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
      <div class="col-md-6"><div class="chart-card">
        <div id="chart-qi-inbound"></div>
        <div class="chart-insight">Queues that send the most cases into the selected queue. If the same upstream queue dominates, that's where routing decisions are consistently wrong.</div>
      </div></div>
      <div class="col-md-6"><div class="chart-card">
        <div id="chart-qi-outbound"></div>
        <div class="chart-insight">Where cases go after leaving this queue. A long tail of varied destinations suggests the queue isn't consistently routing to the right team — it's guessing.</div>
      </div></div>
    </div>
    <div class="row g-3 mt-1">
      <div class="col-md-8"><div class="chart-card">
        <div id="chart-qi-dwell"></div>
        <div class="chart-insight">Distribution of how long cases actually sat in this queue (in days). The median line shows typical wait; P90 shows the worst-case experience for 9 in 10 cases. A wide spread means inconsistent handling.</div>
      </div></div>
      <div class="col-md-4">
        <div class="chart-card" style="height:100%;min-height:300px;">
          <h6 style="font-weight:700;margin-bottom:.5rem;">Top 10 Journey Paths Through Queue <small class="text-muted">(click for cases)</small></h6>
          <div id="qi-paths-table" style="font-size:.78rem;overflow-y:auto;max-height:340px;"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── TAB 6: DATA EXPLORER ── -->
  <div id="tab-explorer" class="tab-panel">
    <div class="guide-stmt">
      <strong>Everything in this report is built from the data below.</strong>
      Browse case-level summaries, filter by transfer count, then download the CSV.
      No black boxes.
    </div>
    <div class="d-flex gap-2 align-items-center mb-2 flex-wrap">
      <div class="filter-label mb-0">View:</div>
      <select id="ex-view" class="form-select form-select-sm" style="width:auto;" onchange="renderExplorer()">
        <option value="cases" selected>Case List</option>
        <option value="journeys">Queue Journey (Raw)</option>
        <option value="breakdown">Transfer Breakdown</option>
        <option value="performance">Queue Performance</option>
      </select>
      <div class="filter-label mb-0 ms-2">Transfer Count:</div>
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

  <!-- ── TAB 8: DEFINITIONS ── -->
  <div id="tab-definitions" class="tab-panel">
    <table class="table data-table table-bordered table-sm" style="font-size:.82rem;background:#fff;">
      <thead>
        <tr>
          <th style="width:22%;background:#1a1a2e;">Term</th>
          <th style="width:52%;background:#1a1a2e;">Definition</th>
          <th style="width:26%;background:#1a1a2e;">Formula / Calculation</th>
        </tr>
      </thead>
      <tbody>
        <!-- ── Case & Queue Concepts ── -->
        <tr style="background:#EBF3FB;">
          <td colspan="3" style="font-weight:800;font-size:.72rem;text-transform:uppercase;
            letter-spacing:.5px;color:#0078D4;padding:6px 10px;border-color:#BDD7F0;">
            Case &amp; Queue Concepts
          </td>
        </tr>
        <tr>
          <td><strong>Case</strong></td>
          <td>A single customer contact opened in Messenger. One case may pass through multiple queues before it is closed. Each case has a unique Case ID and a recorded open and close time.</td>
          <td class="text-muted">—</td>
        </tr>
        <tr>
          <td><strong>Queue</strong></td>
          <td>A work-pool to which a case is assigned. When an agent cannot resolve the case, they transfer it to a different queue. The sequence of queues a case visits forms its <em>journey</em>.</td>
          <td class="text-muted">—</td>
        </tr>
        <tr>
          <td><strong>Transfer</strong></td>
          <td>Every time a case moves from one queue to another counts as one transfer. A case that is opened and closed in the same queue has 0 transfers — a perfect first-touch resolution.</td>
          <td><code>transfers = queues visited − 1</code></td>
        </tr>
        <tr>
          <td><strong>Entry Queue</strong></td>
          <td>The <em>first</em> queue a case enters after being opened in Messenger — the team that receives the initial customer contact and decides how to route it.</td>
          <td class="text-muted">—</td>
        </tr>
        <tr>
          <td><strong>Final Queue</strong></td>
          <td>The <em>last</em> queue a case was in when it was closed — the team that ultimately resolved the customer contact, regardless of transfers before reaching them.</td>
          <td class="text-muted">—</td>
        </tr>
        <tr>
          <td><strong>Intermediary Queue</strong></td>
          <td>Any queue a case passes through that is neither entry nor final. Time spent here is pure routing delay — the customer is waiting but no resolution is being reached.</td>
          <td class="text-muted">—</td>
        </tr>

        <!-- ── Resolution & Routing Metrics ── -->
        <tr style="background:#EBF3FB;">
          <td colspan="3" style="font-weight:800;font-size:.72rem;text-transform:uppercase;
            letter-spacing:.5px;color:#0078D4;padding:6px 10px;border-color:#BDD7F0;">
            Resolution &amp; Routing Metrics
          </td>
        </tr>
        <tr>
          <td><strong>Direct Resolution Rate (DRR) / First-Touch Resolution (FTR)</strong></td>
          <td>The percentage of cases closed by the entry queue without any Messenger transfer. A DRR of 70% means 3 in 10 cases needed at least one transfer.</td>
          <td><code>(cases with 0 transfers ÷ total cases) × 100</code><br><span style="font-size:.75rem;color:#107C10;">Example: 420/600 = 70% DRR</span></td>
        </tr>
        <tr>
          <td><strong>Multi-Transfer Rate</strong></td>
          <td>The percentage of cases transferred <em>two or more times</em>. These represent the most severe mis-routing — bouncing through multiple queues before reaching the right team.</td>
          <td><code>(cases with 2+ transfers ÷ total cases) × 100</code></td>
        </tr>
        <tr>
          <td><strong>Transfer Groups (0 / 1 / 2 / 3+)</strong></td>
          <td>Cases grouped into four buckets for chart comparisons. The 3+ group captures all cases with three or more transfers, which consistently shows the highest cost and effort.</td>
          <td><code>0 = direct &nbsp;| 1 = one &nbsp;| 2 = two &nbsp;| 3+ = three+</code></td>
        </tr>
        <tr>
          <td><strong>Loop / Rework Rate</strong></td>
          <td>A loop occurs when a case visits the same queue more than once — it was sent back to a queue that had already seen it. Every loop is an avoidable transfer.</td>
          <td><code>(cases with repeated queue visit ÷ total) × 100</code><br><span style="font-size:.75rem;color:#107C10;">Example: A → B → A = 1 loop</span></td>
        </tr>
        <tr>
          <td><strong>Routing Days</strong></td>
          <td>Total calendar days a case spent in transit between queues — time in all queues <em>except</em> the final resolving queue. A first-touch case has 0 routing days.</td>
          <td><code>SUM(DAYS_IN_QUEUE) excluding final queue</code></td>
        </tr>
        <tr>
          <td><strong>Dwell Days</strong></td>
          <td>Time a case spent inside one specific queue, from arrival to departure. Shown as median and P90. Long dwell times indicate capacity constraints or process delays within that queue.</td>
          <td><code>DAYS_IN_QUEUE for one queue step</code></td>
        </tr>

        <!-- ── Effort & Cost Metrics ── -->
        <tr style="background:#EBF3FB;">
          <td colspan="3" style="font-weight:800;font-size:.72rem;text-transform:uppercase;
            letter-spacing:.5px;color:#0078D4;padding:6px 10px;border-color:#BDD7F0;">
            Effort &amp; Cost Metrics
          </td>
        </tr>
        <tr>
          <td><strong>Average Handle Time (AHT)</strong></td>
          <td>Total active agent time on a case, in <strong>minutes</strong>. Includes all queue interactions across the journey. AHT rises with each transfer as agents re-read context, re-engage the customer, and re-work prior steps. Median is used as the benchmark — resistant to extreme outliers.</td>
          <td><code>TOTALACTIVEAHT ÷ 60 → minutes</code></td>
        </tr>
        <tr>
          <td><strong>Customer Messages</strong></td>
          <td>Total messages sent by the customer during the case. Higher counts reflect greater customer effort and frustration — re-explaining their query to each new agent they are transferred to.</td>
          <td><code>MESSAGESRECEIVED_CUSTOMER (max across all steps)</code></td>
        </tr>
        <tr>
          <td><strong>P90 (90th Percentile)</strong></td>
          <td>The value below which 90% of cases fall. A large gap between median and P90 means a minority of cases experience significantly worse outcomes — often due to complex routing chains or queue backlogs.</td>
          <td><code>QUANTILE_CONT(metric, 0.9)</code><br><span style="font-size:.75rem;color:#107C10;">Median dwell 1.2d, P90 6.5d = some cases wait much longer</span></td>
        </tr>
        <tr>
          <td><strong>Intermediary Queue Delay (Pareto Distribution)</strong></td>
          <td>The Process &amp; Routing Pareto Distribution ranks queues by total delay days accumulated as intermediary stops — excluding entry and final queues. A small number of queues typically account for the majority of total routing delay.</td>
          <td><code>SUM(DAYS_IN_QUEUE) where queue is not entry or final</code></td>
        </tr>

        <!-- ── Segmentation ── -->
        <tr style="background:#EBF3FB;">
          <td colspan="3" style="font-weight:800;font-size:.72rem;text-transform:uppercase;
            letter-spacing:.5px;color:#0078D4;padding:6px 10px;border-color:#BDD7F0;">
            Segmentation &amp; Classification
          </td>
        </tr>
        <tr>
          <td><strong>Retail Segment</strong></td>
          <td>Cases where the entry queue begins with <strong>HD RTL A</strong> but not <strong>HD RTL A PRT</strong>. Represents personal lines customers contacting Hastings Direct retail insurance teams.</td>
          <td><code>entry_queue LIKE 'HD RTL A%' AND NOT LIKE 'HD RTL A PRT%'</code></td>
        </tr>
        <tr>
          <td><strong>Claims Segment</strong></td>
          <td>All cases that do not meet the Retail criteria — primarily cases entering through Claims queues, including HD RTL A PRT queues. Claims cases tend to be more complex with different transfer patterns.</td>
          <td><code>All cases not classified as Retail</code></td>
        </tr>
        <tr>
          <td><strong>In-Hours (IH) vs Out-of-Hours (OOH)</strong></td>
          <td>Whether the case was created within standard working hours (In-Hours = 1) or outside them (Out-of-Hours = 0). Sourced directly from the operational system. OOH cases may have access to fewer queues, driving higher transfer rates.</td>
          <td><code>INHOURS = 1 → in-hours &nbsp;| INHOURS = 0 → out-of-hours</code></td>
        </tr>
        <tr>
          <td><strong>Day of Week &amp; Hour of Day</strong></td>
          <td>Both derived from the case creation timestamp. Used in the Heatmaps tab to identify which days and times generate the highest volume and transfer rates, enabling targeted staffing decisions.</td>
          <td><code>Day = created_at.dayofweek (0=Mon … 6=Sun)<br>Hour = created_at.hour (0–23)</code></td>
        </tr>
        <tr>
          <td><strong>FTR as Entry Queue %</strong></td>
          <td>For a selected queue in Queue Intelligence: the percentage of cases where <em>that queue was both the entry point AND resolved with 0 transfers</em>. Isolates first-port-of-call performance from cases arriving post-transfer.</td>
          <td><code>(entry = queue AND transfers = 0) ÷ total through queue × 100</code></td>
        </tr>
        <tr>
          <td><strong>Avoidable Transfer (Round Trip)</strong></td>
          <td>A case is flagged as avoidable when its journey starts and ends at the same queue — sent back to where it began. Every round-trip was avoidable at the point of first transfer.</td>
          <td><code>entry_queue = final_queue AND transfers &gt; 0</code></td>
        </tr>

        <!-- ── Statistical Terms ── -->
        <tr style="background:#EBF3FB;">
          <td colspan="3" style="font-weight:800;font-size:.72rem;text-transform:uppercase;
            letter-spacing:.5px;color:#0078D4;padding:6px 10px;border-color:#BDD7F0;">
            Statistical &amp; Chart Terms
          </td>
        </tr>
        <tr>
          <td><strong>Median vs Mean</strong></td>
          <td>Median is preferred throughout for effort metrics (AHT, messages) — it is the middle value when sorted and is not distorted by extreme cases. Mean is shown alongside for completeness.</td>
          <td><span style="font-size:.75rem;color:#107C10;">AHT [10,11,12,13,200] → Median=12, Mean=49. Median better represents typical experience.</span></td>
        </tr>
        <tr>
          <td><strong>Box Plot (IQR)</strong></td>
          <td>Used on Cost &amp; Effort to show the distribution of AHT and messages by transfer group. The box spans Q1–Q3 (middle 50% of cases). The line is the median; the diamond is the mean. Whiskers show the typical range, capped at P95 to suppress extreme outliers.</td>
          <td><code>IQR = Q3 − Q1<br>Whisker max = Q3 + 1.5×IQR (capped at P95)</code></td>
        </tr>
        <tr>
          <td><strong>Pareto Distribution / 80-20 Rule</strong></td>
          <td>The Pareto Distribution on Process &amp; Routing ranks intermediary queues by total delay days, with a cumulative % line. The 80% mark reveals which few queues account for most routing delay — fixing those delivers the greatest overall improvement.</td>
          <td class="text-muted">—</td>
        </tr>
        <tr>
          <td><strong>Sankey Diagram</strong></td>
          <td>Flow diagrams in Journey Pathways showing how cases move between queues. Band width is proportional to case volume. Clicking a band opens a case list for that queue-to-queue transition. Forward = from queue; Backward = to queue.</td>
          <td class="text-muted">—</td>
        </tr>
        <tr>
          <td><strong>Heatmap Normalisation</strong></td>
          <td>The Heatmaps tab shows each metric as a <em>percentage of the day's total</em> rather than raw counts. This removes volume differences between days, showing which hours within each day are disproportionately expensive or transfer-heavy relative to that day's baseline.</td>
          <td><code>(hour metric ÷ day total metric) × 100</code></td>
        </tr>
        <tr>
          <td><strong>Date Filters &amp; Data Scope</strong></td>
          <td>All charts and KPIs respond to the date range, entry queue, segment, and in/out-of-hours filters at the top. The date filter applies to the case <em>creation date</em>. Queue filter limits analysis to cases that <em>entered</em> through that queue.</td>
          <td class="text-muted">—</td>
        </tr>
      </tbody>
    </table>

  </div><!-- /tab-definitions -->

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
const ALL_QUEUES = {queues_json};        // entry queues only (for filter panel)
const ALL_TRANS_QUEUES = {trans_queues_json};  // all queues in transitions (for QI + Journey)
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
let EXPLORER_PAGE = 0;
let EXPLORER_TOTAL = 0;
let EXPLORER_WHERE = '';
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
  const opts = ALL_TRANS_QUEUES.map(q => `<option value="${{q}}">${{q}}</option>`).join('');
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
  else if (tabId === 'explorer') renderExplorer();
}};

window.resetFilters = function() {{
  document.getElementById('f-start').value = MIN_DATE;
  document.getElementById('f-end').value = MAX_DATE;
  const qSel = document.getElementById('f-queue');
  Array.from(qSel.options).forEach(o => o.selected = false);
  Array.from(document.getElementById('f-hours').options).forEach(o => o.selected = true);
  Array.from(document.getElementById('f-segment').options).forEach(o => o.selected = true);
  applyFilters();
}};

window.setSegment = function(seg) {{
  const sSel = document.getElementById('f-segment');
  Array.from(sSel.options).forEach(o => o.selected = (o.value === seg));
  applyFilters();
}};

window.setHours = function(h) {{
  const hSel = document.getElementById('f-hours');
  Array.from(hSel.options).forEach(o => o.selected = (o.value === h));
  applyFilters();
}};

// Date range slider helpers
function updateDateFromSlider(which, idx) {{
  const i = parseInt(idx);
  const m = MONTHS[i];
  if (!m) return;
  const startSlider = document.getElementById('f-start-slider');
  const endSlider   = document.getElementById('f-end-slider');
  if (which === 'start') {{
    if (i > parseInt(endSlider.value)) {{ endSlider.value = i; }}
    document.getElementById('f-start').value = m + '-01';
    document.getElementById('f-end').value = getMonthEnd(MONTHS[parseInt(endSlider.value)]);
  }} else {{
    if (i < parseInt(startSlider.value)) {{ startSlider.value = i; }}
    document.getElementById('f-end').value = getMonthEnd(m);
    document.getElementById('f-start').value = MONTHS[parseInt(startSlider.value)] + '-01';
  }}
  updateSliderLabel();
}}

function getMonthEnd(m) {{
  if (!m) return '';
  const [y, mo] = m.split('-').map(Number);
  const d = new Date(y, mo, 0);
  return m + '-' + String(d.getDate()).padStart(2, '0');
}}

function updateSliderLabel() {{
  const si = parseInt(document.getElementById('f-start-slider')?.value || 0);
  const ei = parseInt(document.getElementById('f-end-slider')?.value || 0);
  const sm = MONTHS[si] || '', em = MONTHS[ei] || '';
  const label = document.getElementById('f-slider-label');
  if (label) label.textContent = sm + ' → ' + em;
}}

// Initialise sliders once MONTHS is available
(function initSliders() {{
  const n = MONTHS.length - 1;
  const ss = document.getElementById('f-start-slider');
  const es = document.getElementById('f-end-slider');
  if (ss && es && n >= 0) {{
    ss.max = n; ss.value = 0;
    es.max = n; es.value = n;
    updateSliderLabel();
  }}
}})();

// ═══════════════════════════════════════════════════════
// KPI HELPERS
// ═══════════════════════════════════════════════════════
function kpiCard(label, value, cls, colCls) {{
  const col = colCls || 'col-6 col-md-3';
  return `<div class="${{col}}">
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
      AVG(CASE WHEN transfers>=2 THEN 1.0 ELSE 0 END)*100 as multi_rate,
      AVG(loop_flag)*100 as loop_rate
    FROM cases ${{w}}`);
  const d = rows[0] || {{}};
  const n = (d.total || 0).toLocaleString();
  document.getElementById('case-count-badge').textContent = n + ' cases';
  document.getElementById('overview-kpis').innerHTML = [
    kpiCard('Total Cases', n, 'kpi-primary'),
    kpiCard('Direct Resolution Rate (No Messenger Transfer)', (d.drr||0).toFixed(1)+'%', 'kpi-success', 'col-6 col-md-5'),
    kpiCard('Multi-Transfer Rate', (d.multi_rate||0).toFixed(1)+'%', 'kpi-danger', 'col-6 col-md-2'),
    kpiCard('Loop Rate', (d.loop_rate||0).toFixed(1)+'%', 'kpi-info', 'col-6 col-md-2'),
  ].join('');

  // Journey Pathways is embedded in Overview tab — render it too
  renderJourney();
}}

// ═══════════════════════════════════════════════════════
// TAB 2: PROCESS & ROUTING
// ═══════════════════════════════════════════════════════
async function renderProcess(f) {{
  const w = buildWhere(f, 'c');
  const stats = await q(`SELECT COUNT(*) as total,
    AVG(CASE WHEN transfers=0 THEN 1.0 ELSE 0.0 END)*100 as drr,
    AVG(CASE WHEN loop_flag>0 THEN 1.0 ELSE 0.0 END)*100 as loop_rate,
    COUNT(*) FILTER (WHERE loop_flag > 0) as rework_cases,
    AVG(CASE WHEN transfers>=2 THEN 1.0 ELSE 0.0 END)*100 as multi_rate
    FROM cases c ${{w}}`);
  const d = stats[0] || {{}};
  document.getElementById('process-kpis').innerHTML = [
    kpiCard('Direct Resolution Rate', (d.drr||0).toFixed(1)+'%', 'kpi-success'),
    kpiCard('Loop / Rework Rate', (d.loop_rate||0).toFixed(1)+'%', 'kpi-danger'),
    kpiCard('Cases with Rework', Math.round(Number(d.rework_cases)||0).toLocaleString(), 'kpi-warning'),
    kpiCard('Multi-Transfer Rate', (d.multi_rate||0).toFixed(1)+'%', 'kpi-info'),
  ].join('');

  // Intermediary queues Pareto
  const where_cases = buildWhere(f, 'c');
  const pareto = await q(`
    WITH max_orders AS (
      SELECT CASE_ID, MAX(QUEUE_ORDER) as max_ord FROM transitions GROUP BY CASE_ID
    )
    SELECT t.QUEUE_NEW, SUM(t.DAYS_IN_QUEUE) as delay_days
    FROM transitions t
    JOIN max_orders m ON t.CASE_ID = m.CASE_ID
    WHERE t.QUEUE_ORDER > 1 AND t.QUEUE_ORDER < m.max_ord
      AND t.CASE_ID IN (SELECT CASE_ID FROM cases c ${{where_cases}})
    GROUP BY t.QUEUE_NEW ORDER BY delay_days DESC LIMIT 15`);
  const totalDelay = pareto.reduce((s,r)=>s+r.delay_days,0);
  if (!pareto.length || totalDelay === 0) {{
    Plotly.react('chart-pareto', [], {{title:'No intermediary queue data for this filter',height:420,
      paper_bgcolor:'transparent',plot_bgcolor:'transparent'}}, {{responsive:true}});
  }} else {{
  let cumPct = 0;
  const cumPcts = pareto.map(r => {{ cumPct += r.delay_days/totalDelay*100; return Math.round(cumPct*10)/10; }});
  Plotly.react('chart-pareto', [
    {{
      type:'bar', x:pareto.map(r=>r.QUEUE_NEW), y:pareto.map(r=>r.delay_days),
      marker:{{color:COLORS.danger}},
      text:pareto.map(r=>r.delay_days.toFixed(1)+' days'), textposition:'outside',
      name:'Delay Days',
    }},
    {{
      type:'scatter', mode:'lines+markers',
      x:pareto.map(r=>r.QUEUE_NEW), y:cumPcts,
      yaxis:'y2', name:'Cumulative %',
      line:{{color:COLORS.primary,width:2}}, marker:{{color:COLORS.primary,size:6}},
    }},
  ], {{
    title:'Top Intermediary Queues by Total Delay Days (Pareto Distribution)', height:420,
    margin:{{t:50,l:60,r:60,b:120}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    xaxis:{{tickangle:-35,tickfont:{{size:9}},showgrid:false}},
    yaxis:{{title:'Total Delay Days',showgrid:true,gridcolor:'#EDEBE9'}},
    yaxis2:{{title:'Cumulative %',overlaying:'y',side:'right',range:[0,105],showgrid:false}},
    legend:{{orientation:'h',y:1.1}}, showlegend:true,
  }}, {{responsive:true}});
  }} // end else (pareto has data)

  const entry = await q(`
    SELECT entry_queue,
      AVG(CASE WHEN transfers=0 THEN 100.0 ELSE 0.0 END) as ftr_pct,
      AVG(CASE WHEN transfers>0 THEN 100.0 ELSE 0.0 END) as xfer_pct,
      COUNT(*) as n
    FROM cases c ${{where_cases}}
    GROUP BY entry_queue HAVING COUNT(*) >= 5
    ORDER BY ftr_pct ASC LIMIT 12`);
  Plotly.react('chart-entry-dist', [
    {{
      type:'bar', orientation:'h',
      x:entry.map(r=>Math.round(r.xfer_pct*10)/10),
      y:entry.map(r=>r.entry_queue),
      name:'Transfer %', marker:{{color:COLORS.danger}},
      text:entry.map(r=>Math.round(r.xfer_pct)+'%'), textposition:'inside',
    }},
    {{
      type:'bar', orientation:'h',
      x:entry.map(r=>Math.round(r.ftr_pct*10)/10),
      y:entry.map(r=>r.entry_queue),
      name:'FTR %', marker:{{color:COLORS.success}},
      text:entry.map(r=>Math.round(r.ftr_pct)+'%'), textposition:'inside',
    }},
  ], {{
    title:'Entry Queue FTR vs Transfer % (worst → best)', height:420,
    barmode:'stack',
    margin:{{t:50,l:200,r:60,b:40}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    xaxis:{{range:[0,100],title:'%',showgrid:true,gridcolor:'#EDEBE9'}},
    yaxis:{{tickfont:{{size:9}}}},
    legend:{{orientation:'h',y:1.1}},
  }}, {{responsive:true}});
}}

// ═══════════════════════════════════════════════════════
// TAB 3: COST & EFFORT
// ═══════════════════════════════════════════════════════
let COST_BIN_CASES = {{}};   // transfer_bin -> [case_ids]
let COST_TRACE_BINS = [];   // ordered list of bins that got a trace (for curveNumber mapping)

async function renderCost(f) {{
  const w = buildWhere(f);

  // Fetch raw per-case values in one query.
  // Using raw data (not pre-aggregated) so Plotly computes its own quartiles.
  // Pre-aggregated mode had a silent bug: QUANTILE_CONT returns null for small
  // bins → Math.min(value, null)=0 → upperfence<lowerfence → Plotly drops trace.
  const rows = await q(`
    SELECT CASE WHEN transfers=0 THEN '0' WHEN transfers=1 THEN '1' WHEN transfers=2 THEN '2' ELSE '3+' END as transfer_bin,
      CAST(CASE_ID AS VARCHAR) as cid,
      total_active_aht, messages
    FROM cases ${{w}}`);

  const bins = ['0','1','2','3+'];
  const ahtByBin = {{'0':[],'1':[],'2':[],'3+':[]}};
  const msgByBin = {{'0':[],'1':[],'2':[],'3+':[]}};
  COST_BIN_CASES = {{}};
  for (const r of rows) {{
    const b = r.transfer_bin;
    if (!b) continue;
    if (r.total_active_aht != null) ahtByBin[b].push(Number(r.total_active_aht));
    if (r.messages != null) msgByBin[b].push(Number(r.messages));
    if (!COST_BIN_CASES[b]) COST_BIN_CASES[b] = [];
    COST_BIN_CASES[b].push(r.cid);
  }}
  COST_TRACE_BINS = bins.filter(b => ahtByBin[b].length > 0);

  // JS stat helpers — no SQL null risk
  const jsSort = arr => [...arr].sort((a,b)=>a-b);
  const jsQ1   = arr => {{ const s=jsSort(arr); return s[Math.floor(s.length/4)]; }};
  const jsMed  = arr => {{ if(!arr.length)return 0; const s=jsSort(arr),m=Math.floor(s.length/2); return s.length%2?s[m]:(s[m-1]+s[m])/2; }};
  const jsQ3   = arr => {{ const s=jsSort(arr); return s[Math.floor(s.length*3/4)]; }};
  const jsMean = arr => arr.reduce((s,v)=>s+v,0)/arr.length;
  const jsP95  = arr => {{ const s=jsSort(arr); return s[Math.min(Math.floor(s.length*0.95), s.length-1)]; }};

  const base_aht = jsMed(ahtByBin['0']), high_aht = jsMed(ahtByBin['3+']);
  const base_msg = jsMed(msgByBin['0']), high_msg = jsMed(msgByBin['3+']);
  const aht_pct = base_aht > 0 ? (high_aht/base_aht - 1)*100 : 0;
  const msg_pct = base_msg > 0 ? (high_msg/base_msg - 1)*100 : 0;

  document.getElementById('cost-guide-stmt').innerHTML =
    `Every transfer doesn't just delay the customer, <strong>it inflates the total effort.</strong>
     A case that gets transferred 3+ times costs <strong>${{Math.round(aht_pct)}}% more handle time</strong>
     and generates <strong>${{Math.round(msg_pct)}}% more customer messages</strong>
     than one resolved first-touch. <strong>This is the compounding cost of mis-routing.</strong>`;

  document.getElementById('cost-kpis').innerHTML = [
    kpiCard('AHT — First Touch', Math.round(base_aht)+' min', 'kpi-success'),
    kpiCard('AHT — 3+ Transfers', Math.round(high_aht)+' min', 'kpi-danger'),
    kpiCard('Messages — First Touch', Math.round(base_msg), 'kpi-success'),
    kpiCard('Messages — 3+ Transfers', Math.round(high_msg), 'kpi-danger'),
  ].join('');

  document.getElementById('cost-insight').innerHTML =
    `Every additional transfer inflates handle time by ~${{Math.round(aht_pct/3)}}% per step
     and customer messages by ~${{Math.round(msg_pct/3)}}% per step.`;

  // Raw-data box traces with P95 cap to tame extreme outliers.
  // Values above the P95 for each bin are excluded before being passed to Plotly.
  // Plotly then computes quartiles + whiskers from the filtered arrays itself.
  // Annotations label median & mean for quick comparison.
  const ahtTraces = [], msgTraces = [];
  const ahtAnnotations = [], msgAnnotations = [];

  for (const bin of bins) {{
    if (!ahtByBin[bin].length) continue;
    const label = bin + (bin==='1'?' transfer':' transfers');
    const color = BIN_COLORS[bin];

    // Filter each bin to values ≤ P95 to remove extreme tail outliers
    const p95a = jsP95(ahtByBin[bin]);
    const filtAht = ahtByBin[bin].filter(v => v <= p95a);
    const p95m = jsP95(msgByBin[bin]);
    const filtMsg = msgByBin[bin].filter(v => v <= p95m);

    const amed=jsMed(filtAht), amean=filtAht.length?jsMean(filtAht):0;
    const aq3=jsQ3(filtAht);
    const mmed=jsMed(filtMsg), mmean=filtMsg.length?jsMean(filtMsg):0;
    const mq3=jsQ3(filtMsg);

    // Pass raw (filtered) y arrays — Plotly computes quartiles + whiskers
    ahtTraces.push({{
      type:'box', name:label,
      y:filtAht, boxmean:true,
      fillcolor:color+'55', line:{{color}},
      marker:{{color, opacity:0.5, size:3}},
    }});
    msgTraces.push({{
      type:'box', name:label,
      y:filtMsg, boxmean:true,
      fillcolor:color+'55', line:{{color}},
      marker:{{color, opacity:0.5, size:3}},
    }});

    // Annotations above each box
    const aOffset = Math.max(aq3*0.08, 1);
    ahtAnnotations.push({{
      x:label, y:aq3+aOffset, yanchor:'bottom',
      text:`Med <b>${{Math.round(amed)}}</b> | Mean <b>${{Math.round(amean)}}</b>`,
      showarrow:false, font:{{size:9,color:'#333'}},
      bgcolor:'rgba(255,255,255,0.88)', borderpad:3,
    }});
    const mOffset = Math.max(mq3*0.08, 0.5);
    msgAnnotations.push({{
      x:label, y:mq3+mOffset, yanchor:'bottom',
      text:`Med <b>${{Math.round(mmed)}}</b> | Mean <b>${{Math.round(mmean)}}</b>`,
      showarrow:false, font:{{size:9,color:'#333'}},
      bgcolor:'rgba(255,255,255,0.88)', borderpad:3,
    }});
  }}

  const p95Note = {{
    xref:'paper', yref:'paper', x:1, y:-0.07,
    text:'<i>Values above the 95th percentile per group excluded for visual clarity</i>',
    showarrow:false, font:{{size:8.5, color:'#888'}}, xanchor:'right',
  }};

  const boxLayout = (title, ytitle, annots) => ({{
    title, height:420, showlegend:false,
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    yaxis:{{title:ytitle, showgrid:true, gridcolor:'#EDEBE9', rangemode:'tozero'}},
    xaxis:{{showgrid:false}}, margin:{{t:50,l:60,r:30,b:55}},
    annotations: [...annots, p95Note],
  }});

  const ahtDiv = document.getElementById('chart-aht-box');
  Plotly.react(ahtDiv, ahtTraces, boxLayout('Handle Time by Transfer Count','AHT (min)', ahtAnnotations), {{responsive:true}});
  ahtDiv.on('plotly_click', data => showCostModal(data, 'AHT'));

  const msgDiv = document.getElementById('chart-msg-box');
  Plotly.react(msgDiv, msgTraces, boxLayout('Customer Messages by Transfer Count','Messages', msgAnnotations), {{responsive:true}});
  msgDiv.on('plotly_click', data => showCostModal(data, 'Messages'));

}}

async function showCostModal(data, chartType) {{
  const curveIdx = data.points[0].curveNumber;
  const bin = COST_TRACE_BINS[curveIdx];
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

  // Fetch heatmap data — % of day
  const hmRows = await q(`
    WITH raw AS (
      SELECT day_of_week, hour_of_day,
        COUNT(*) as volume,
        MEDIAN(total_active_aht) as med_aht,
        MEDIAN(messages) as med_msgs,
        MEDIAN(routing_days) as med_routing,
        AVG(CAST(inhours AS DOUBLE)) as inhours_rate
      FROM cases ${{w}} AND day_of_week IS NOT NULL AND hour_of_day IS NOT NULL
      GROUP BY day_of_week, hour_of_day
    ),
    day_sums AS (
      SELECT day_of_week, SUM(volume) as dv,
        SUM(med_aht) as da, SUM(med_msgs) as dm, SUM(med_routing) as dr
      FROM raw GROUP BY day_of_week
    )
    SELECT r.day_of_week, r.hour_of_day,
      CASE WHEN d.dv>0 THEN r.volume*100.0/d.dv ELSE 0 END as volume_pct,
      CASE WHEN d.da>0 THEN r.med_aht*100.0/d.da ELSE 0 END as aht_pct,
      CASE WHEN d.dm>0 THEN r.med_msgs*100.0/d.dm ELSE 0 END as msgs_pct,
      CASE WHEN d.dr>0 THEN r.med_routing*100.0/d.dr ELSE 0 END as routing_pct,
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
    colorbar:{{title:{{text:HEATMAP_VIEW==='inhours'?'In-Hours Rate (%)':'% of day',font:{{size:10}}}},thickness:12,len:0.85}},
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
  const qSafe = selQ.replace(/'/g, "''");

  // Main case-level stats + dwell from transitions
  const [stats, dwellStats] = await Promise.all([
    q(`SELECT COUNT(*) as n,
        AVG(CASE WHEN c.entry_queue='${{qSafe}}' AND c.transfers=0 THEN 1.0 ELSE 0.0 END)*100 as ftr_as_entry_pct,
        SUM(CASE WHEN c.entry_queue='${{qSafe}}' THEN 1 ELSE 0 END) as entry_count
      FROM cases c ${{w}} AND CAST(c.CASE_ID AS VARCHAR) IN (
        SELECT CAST(CASE_ID AS VARCHAR) FROM transitions WHERE QUEUE_NEW='${{qSafe}}'
      )`),
    q(`SELECT COUNT(*) as n,
        MEDIAN(DAYS_IN_QUEUE) as med_dwell,
        QUANTILE_CONT(DAYS_IN_QUEUE, 0.9) as p90_dwell,
        SUM(DAYS_IN_QUEUE) as total_dwell
      FROM transitions
      WHERE QUEUE_NEW='${{qSafe}}'
        AND CASE_ID IN (SELECT CASE_ID FROM cases c ${{w}})`),
  ]);

  const d  = stats[0] || {{}};
  const dw = dwellStats[0] || {{}};

  document.getElementById('qi-kpis').innerHTML = [
    kpiCard('Cases Through Queue', (d.n||0).toLocaleString(), 'kpi-primary'),
    kpiCard('Median Dwell Days', (dw.med_dwell||0).toFixed(1), 'kpi-info'),
    kpiCard('P90 Dwell Days', (dw.p90_dwell||0).toFixed(1), 'kpi-warning'),
    kpiCard('FTR as Entry Queue', (d.ftr_as_entry_pct||0).toFixed(1)+'%', 'kpi-success'),
  ].join('');

  const inbound = await q(`
    SELECT prev.QUEUE_NEW as from_queue, COUNT(*) as n
    FROM transitions t
    JOIN transitions prev ON t.CASE_ID=prev.CASE_ID AND t.QUEUE_ORDER=prev.QUEUE_ORDER+1
    WHERE t.QUEUE_NEW='${{qSafe}}'
      AND t.CASE_ID IN (SELECT CASE_ID FROM cases c ${{w}})
    GROUP BY prev.QUEUE_NEW ORDER BY n DESC LIMIT 12`);

  const outbound = await q(`
    SELECT nxt.QUEUE_NEW as to_queue, COUNT(*) as n
    FROM transitions t
    JOIN transitions nxt ON t.CASE_ID=nxt.CASE_ID AND nxt.QUEUE_ORDER=t.QUEUE_ORDER+1
    WHERE t.QUEUE_NEW='${{qSafe}}'
      AND t.CASE_ID IN (SELECT CASE_ID FROM cases c ${{w}})
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

  // Dwell histogram
  const dwellRows = await q(`
    SELECT CAST(DAYS_IN_QUEUE AS DOUBLE) as d
    FROM transitions
    WHERE QUEUE_NEW='${{qSafe}}'
      AND CASE_ID IN (SELECT CASE_ID FROM cases c ${{w}})
      AND DAYS_IN_QUEUE IS NOT NULL`);
  const dwellVals = dwellRows.map(r=>r.d);
  const medDwell = dw.med_dwell || 0;
  const p90Dwell = dw.p90_dwell || 0;
  Plotly.react('chart-qi-dwell', [
    {{type:'histogram', x:dwellVals, nbinsx:20,
      marker:{{color:COLORS.primary+'99', line:{{color:COLORS.primary, width:1}}}},
      name:'Cases'}},
  ], {{
    title:`Dwell Time Distribution in ${{selQ}} (days)`, height:320,
    margin:{{t:50,l:55,r:20,b:45}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    xaxis:{{title:'Days in Queue', showgrid:true, gridcolor:'#EDEBE9'}},
    yaxis:{{title:'Cases', showgrid:true, gridcolor:'#EDEBE9'}},
    shapes:[
      {{type:'line', x0:medDwell, x1:medDwell, y0:0, y1:1, yref:'paper',
        line:{{color:'#107C10', width:2, dash:'dash'}}}},
      {{type:'line', x0:p90Dwell, x1:p90Dwell, y0:0, y1:1, yref:'paper',
        line:{{color:'#D83B01', width:2, dash:'dot'}}}},
    ],
    annotations:[
      {{x:medDwell, y:0.95, yref:'paper', text:`Median: ${{medDwell.toFixed(1)}}d`,
        showarrow:false, font:{{size:10, color:'#107C10'}}, xanchor:'left', xshift:4}},
      {{x:p90Dwell, y:0.82, yref:'paper', text:`P90: ${{p90Dwell.toFixed(1)}}d`,
        showarrow:false, font:{{size:10, color:'#D83B01'}}, xanchor:'left', xshift:4}},
    ],
  }}, {{responsive:true}});

  // Top 10 journey paths through selected queue (no LIST() — fetch case IDs on click)
  const pathRows = await q(`
    WITH case_paths AS (
      SELECT CASE_ID,
        STRING_AGG(QUEUE_NEW, ' → ' ORDER BY QUEUE_ORDER) as full_path
      FROM transitions
      WHERE CASE_ID IN (
        SELECT DISTINCT CASE_ID FROM transitions WHERE QUEUE_NEW='${{qSafe}}'
          AND CASE_ID IN (SELECT CASE_ID FROM cases c ${{w}})
      )
      GROUP BY CASE_ID
    )
    SELECT full_path, COUNT(*) as n
    FROM case_paths
    GROUP BY full_path ORDER BY n DESC LIMIT 10`);

  // Build table via DOM to avoid any quote-escaping issues in onclick handlers
  const pathContainer = document.getElementById('qi-paths-table');
  if (!pathRows.length) {{
    pathContainer.innerHTML = '<p class="text-muted">No data.</p>';
  }} else {{
    const tbl = document.createElement('table');
    tbl.className = 'table table-sm';
    tbl.style.fontSize = '.75rem';
    tbl.innerHTML = '<thead><tr><th>Path</th><th style="text-align:right">Cases</th></tr></thead>';
    const tbody = document.createElement('tbody');
    for (const row of pathRows) {{
      const fp = row.full_path;
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
      tr.innerHTML = `<td style="word-break:break-word;">${{fp}}</td><td style="text-align:right;font-weight:600;">${{row.n}}</td>`;
      tr.addEventListener('click', async () => {{
        const fpSafe = fp.replace(/'/g, "''");
        const caseRows = await q(`
          WITH cp AS (
            SELECT CASE_ID, STRING_AGG(QUEUE_NEW,' → ' ORDER BY QUEUE_ORDER) as full_path
            FROM transitions
            WHERE CASE_ID IN (SELECT DISTINCT CASE_ID FROM transitions WHERE QUEUE_NEW='${{qSafe}}'
              AND CASE_ID IN (SELECT CASE_ID FROM cases c ${{w}}))
            GROUP BY CASE_ID
          )
          SELECT CAST(CASE_ID AS VARCHAR) as cid FROM cp WHERE full_path='${{fpSafe}}'`);
        showCaseModal('Path: ' + fp, caseRows.map(r => r.cid));
      }});
      tbody.appendChild(tr);
    }}
    tbl.appendChild(tbody);
    pathContainer.innerHTML = '';
    pathContainer.appendChild(tbl);
  }}
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
  const allPathCids = pathCounts.flatMap(p=>p.cids).map(c=>`'${{String(c).replace(/'/g,"''")}}'`).join(',');
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

  // Convert hex → rgba so link bands are clearly visible against the white background
  function hexRgba(hex, a) {{
    const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
    return `rgba(${{r}},${{g}},${{b}},${{a}})`;
  }}

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
      color:links.map(l=>hexRgba(CHART_COLORS[nodeIdx[l.s]%CHART_COLORS.length], 0.45)),
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
window.renderExplorer = async function(page) {{
  page = page ?? 0;
  const f = getFilterState();
  let w = buildWhere(f);
  const xfer = document.getElementById('ex-xfer').value;
  if      (xfer === '0')  w += ` AND CAST(transfers AS INT) = 0`;
  else if (xfer === '1')  w += ` AND CAST(transfers AS INT) = 1`;
  else if (xfer === '2')  w += ` AND CAST(transfers AS INT) = 2`;
  else if (xfer === '3+') w += ` AND CAST(transfers AS INT) >= 3`;
  EXPLORER_WHERE = w;
  EXPLORER_PAGE = page;

  const view = document.getElementById('ex-view').value;

  // ── Case List (default, paginated) ──────────────────────────────────────
  if (view === 'cases') {{
    const countRow = await q(`SELECT COUNT(*) as n FROM cases ${{w}}`);
    EXPLORER_TOTAL = countRow[0]?.n || 0;
    document.getElementById('ex-count').textContent = EXPLORER_TOTAL.toLocaleString() + ' cases';

    const offset = page * PAGE_SIZE;
    const rows = await q(`
      SELECT CAST(CASE_ID AS VARCHAR) as case_id, entry_queue, final_queue,
        CAST(transfers AS INT) as transfers,
        CAST(transfer_bin AS VARCHAR) as transfer_bin,
        ROUND(total_active_aht, 0) as aht_min,
        ROUND(routing_days, 1) as routing_days,
        CAST(messages AS INT) as messages,
        segment,
        CAST(inhours AS INT) as inhours,
        CAST(loop_flag AS INT) as loop_flag
      FROM cases ${{w}} ORDER BY transfers DESC, total_active_aht DESC
      LIMIT ${{PAGE_SIZE}} OFFSET ${{offset}}`);

    const cols = ['case_id','entry_queue','final_queue','transfers','transfer_bin',
                  'aht_min','routing_days','messages','segment','inhours','loop_flag'];
    const headers = ['Case ID','Entry Queue','Final Queue','# Transfers (exact)','Chart Group (3+=3 or more)',
                     'AHT (min)','Routing Days','Messages','Segment','In-Hours','Loop'];

    const totalPages = Math.ceil(EXPLORER_TOTAL / PAGE_SIZE);
    let html = `<div style="overflow-x:auto">
      <p style="font-size:.75rem;color:#888;margin-bottom:.25rem;">
        <strong># Transfers (exact)</strong> shows the real transfer count per case.
        Charts on other tabs group 3, 4, 5… into a single <strong>3+</strong> bucket — so cases showing 4 or 5 here appear in the "3+" bars elsewhere.
      </p>
      <table class="table data-table table-sm">
      <thead><tr>${{headers.map(h=>`<th>${{h}}</th>`).join('')}}</tr></thead><tbody>`;
    for (const row of rows) {{
      html += `<tr onclick="showCaseModal('Case ${{row.case_id}}',${{JSON.stringify([String(row.case_id)])}})">
        ${{cols.map(c=>`<td>${{row[c]??''}}</td>`).join('')}}</tr>`;
    }}
    html += '</tbody></table></div>';
    document.getElementById('explorer-table-container').innerHTML = html;

    let pagerHtml = `<button onclick="window.renderExplorer(${{EXPLORER_PAGE-1}})" ${{EXPLORER_PAGE===0?'disabled':''}}>‹</button>`;
    const sp = Math.max(0, EXPLORER_PAGE-2), ep = Math.min(totalPages, sp+5);
    for (let i=sp; i<ep; i++) {{
      pagerHtml += `<button class="${{i===EXPLORER_PAGE?'active':''}}" onclick="window.renderExplorer(${{i}})">${{i+1}}</button>`;
    }}
    pagerHtml += `<button onclick="window.renderExplorer(${{EXPLORER_PAGE+1}})" ${{EXPLORER_PAGE>=totalPages-1?'disabled':''}}>›</button>`;
    pagerHtml += ` <span style="font-size:.75rem;color:#888">Page ${{EXPLORER_PAGE+1}} of ${{totalPages}} (${{EXPLORER_TOTAL.toLocaleString()}} total)</span>`;
    document.getElementById('explorer-pager').innerHTML = pagerHtml;
    return;
  }}

  // ── Queue Journey Raw ────────────────────────────────────────────────────
  if (view === 'journeys') {{
    const countRow = await q(`SELECT COUNT(*) as n FROM transitions
      WHERE CASE_ID IN (SELECT CASE_ID FROM cases ${{w}})`);
    EXPLORER_TOTAL = countRow[0]?.n || 0;
    document.getElementById('ex-count').textContent = EXPLORER_TOTAL.toLocaleString() + ' transitions';

    const offset = page * PAGE_SIZE;
    const rows = await q(`
      SELECT CAST(t.CASE_ID AS VARCHAR) as case_id,
        CAST(t.QUEUE_ORDER AS INT) as queue_order,
        t.QUEUE_NEW as queue,
        ROUND(t.DAYS_IN_QUEUE, 2) as days_in_queue,
        c.entry_queue, c.final_queue,
        CAST(c.transfers AS INT) as transfers,
        c.segment
      FROM transitions t
      JOIN cases c ON t.CASE_ID = c.CASE_ID
      WHERE c.CASE_ID IN (SELECT CASE_ID FROM cases ${{w}})
      ORDER BY t.CASE_ID, t.QUEUE_ORDER
      LIMIT ${{PAGE_SIZE}} OFFSET ${{offset}}`);

    const cols = ['case_id','queue_order','queue','days_in_queue','entry_queue','final_queue','transfers','segment'];
    const headers = ['Case ID','Step','Queue','Days in Queue','Entry Queue','Final Queue','Transfers','Segment'];
    const totalPages = Math.ceil(EXPLORER_TOTAL / PAGE_SIZE);
    let html = `<div style="overflow-x:auto"><table class="table data-table table-sm">
      <thead><tr>${{headers.map(h=>`<th>${{h}}</th>`).join('')}}</tr></thead><tbody>`;
    for (const row of rows) {{
      html += `<tr onclick="showCaseModal('Case ${{row.case_id}}',${{JSON.stringify([String(row.case_id)])}})">
        ${{cols.map(c=>`<td>${{row[c]??''}}</td>`).join('')}}</tr>`;
    }}
    html += '</tbody></table></div>';
    document.getElementById('explorer-table-container').innerHTML = html;

    let pagerHtml = `<button onclick="window.renderExplorer(${{EXPLORER_PAGE-1}})" ${{EXPLORER_PAGE===0?'disabled':''}}>‹</button>`;
    const sp = Math.max(0, EXPLORER_PAGE-2), ep = Math.min(totalPages, sp+5);
    for (let i=sp; i<ep; i++) {{
      pagerHtml += `<button class="${{i===EXPLORER_PAGE?'active':''}}" onclick="window.renderExplorer(${{i}})">${{i+1}}</button>`;
    }}
    pagerHtml += `<button onclick="window.renderExplorer(${{EXPLORER_PAGE+1}})" ${{EXPLORER_PAGE>=totalPages-1?'disabled':''}}>›</button>`;
    pagerHtml += ` <span style="font-size:.75rem;color:#888">Page ${{EXPLORER_PAGE+1}} of ${{totalPages}} (${{EXPLORER_TOTAL.toLocaleString()}} total)</span>`;
    document.getElementById('explorer-pager').innerHTML = pagerHtml;
    return;
  }}

  // ── Transfer Breakdown (aggregated, no pagination) ───────────────────────
  if (view === 'breakdown') {{
    const rows = await q(`
      SELECT CASE WHEN transfers=0 THEN '0' WHEN transfers=1 THEN '1' WHEN transfers=2 THEN '2' ELSE '3+' END as transfer_bin,
        COUNT(*) as cases,
        MEDIAN(total_active_aht) as med_aht,
        ROUND(AVG(total_active_aht),1) as mean_aht,
        MEDIAN(messages) as med_msg,
        ROUND(AVG(messages),1) as mean_msg,
        ROUND(AVG(routing_days),2) as avg_routing,
        ROUND(AVG(CASE WHEN loop_flag>0 THEN 1.0 ELSE 0.0 END)*100,1) as loop_pct,
        ROUND(AVG(CASE WHEN inhours=1 THEN 1.0 ELSE 0.0 END)*100,1) as inhours_pct
      FROM cases ${{w}}
      GROUP BY 1 ORDER BY MIN(transfers)`);

    const cols = ['transfer_bin','cases','med_aht','mean_aht','med_msg','mean_msg','avg_routing','loop_pct','inhours_pct'];
    const headers = ['Group','Cases','Median AHT (min)','Mean AHT (min)','Median Messages','Mean Messages','Avg Routing Days','Loop Rate %','In-Hours %'];

    document.getElementById('ex-count').textContent = rows.length + ' groups';
    // Build using DOM to avoid inline onclick quote escaping
    const brkTbl = document.createElement('table');
    brkTbl.className = 'table data-table table-sm';
    brkTbl.innerHTML = `<thead><tr>${{headers.map(h=>`<th>${{h}}</th>`).join('')}}</tr></thead>`;
    const brkBody = document.createElement('tbody');
    for (const row of rows) {{
      const bin = row.transfer_bin;
      const transferFilter = bin==='0'?'CAST(transfers AS INT)=0':bin==='1'?'CAST(transfers AS INT)=1':bin==='2'?'CAST(transfers AS INT)=2':'CAST(transfers AS INT)>=3';
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
      tr.innerHTML = cols.map(c=>`<td>${{row[c]??''}}</td>`).join('');
      tr.addEventListener('click', async () => {{
        const r = await q(`SELECT CAST(CASE_ID AS VARCHAR) as cid FROM cases ${{w}} AND ${{transferFilter}}`);
        showCaseModal('Group '+bin, r.map(x=>x.cid));
      }});
      brkBody.appendChild(tr);
    }}
    brkTbl.appendChild(brkBody);
    const brkContainer = document.getElementById('explorer-table-container');
    brkContainer.innerHTML = '<div style="overflow-x:auto"></div>';
    brkContainer.firstChild.appendChild(brkTbl);
    html += '</tbody></table></div>';
    document.getElementById('explorer-table-container').innerHTML = html;
    document.getElementById('explorer-pager').innerHTML = '';
    return;
  }}

  // ── Queue Performance (aggregated by entry queue) ────────────────────────
  if (view === 'performance') {{
    const rows = await q(`
      SELECT entry_queue,
        COUNT(*) as cases,
        ROUND(AVG(CASE WHEN transfers=0 THEN 100.0 ELSE 0.0 END),1) as ftr_pct,
        ROUND(AVG(transfers),2) as avg_xfer,
        MEDIAN(total_active_aht) as med_aht,
        MEDIAN(routing_days) as med_routing,
        ROUND(AVG(CAST(loop_flag AS DOUBLE))*100,1) as loop_pct,
        ROUND(AVG(CASE WHEN inhours=1 THEN 1.0 ELSE 0.0 END)*100,1) as inhours_pct
      FROM cases ${{w}}
      GROUP BY entry_queue ORDER BY cases DESC`);

    const cols = ['entry_queue','cases','ftr_pct','avg_xfer','med_aht','med_routing','loop_pct','inhours_pct'];
    const headers = ['Entry Queue','Cases','FTR %','Avg Transfers','Median AHT (min)','Median Routing Days','Loop Rate %','In-Hours %'];

    document.getElementById('ex-count').textContent = rows.length + ' queues';
    let html = `<div style="overflow-x:auto"><table class="table data-table table-sm">
      <thead><tr>${{headers.map(h=>`<th>${{h}}</th>`).join('')}}</tr></thead><tbody>`;
    for (const row of rows) {{
      html += `<tr>
        ${{cols.map(c=>`<td>${{row[c]??''}}</td>`).join('')}}</tr>`;
    }}
    html += '</tbody></table></div>';
    document.getElementById('explorer-table-container').innerHTML = html;
    document.getElementById('explorer-pager').innerHTML = '';
  }}
}};

window.downloadCSV = async function() {{
  const rows = await q(`
    SELECT CAST(CASE_ID AS VARCHAR) as case_id, entry_queue, final_queue,
      CAST(transfers AS INT) as transfers,
      CAST(transfer_bin AS VARCHAR) as transfer_bin,
      ROUND(total_active_aht, 0) as aht_min,
      ROUND(routing_days, 1) as routing_days,
      CAST(messages AS INT) as messages,
      segment,
      CAST(inhours AS INT) as inhours,
      CAST(loop_flag AS INT) as loop_flag
    FROM cases ${{EXPLORER_WHERE}} ORDER BY transfers DESC, total_active_aht DESC`);
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const csv = [cols.join(','), ...rows.map(r=>cols.map(c=>JSON.stringify(r[c]??'')).join(','))].join('\\n');
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

<!-- FIXED 4Cs CALLOUT (visible on all tabs) -->
<div id="fcs-callout">
  ⭐ Found this useful?
  <a id="fcs-link" href="#" target="_blank" rel="noopener"
    style="font-weight:700;color:#B8860B;text-decoration:none;border-bottom:1px solid #B8860B;">
    Submit a 4Cs nomination
  </a>
</div>

<!-- FOOTER -->
<div style="background:#1a1a2e;color:rgba(255,255,255,.45);font-size:.72rem;
  text-align:center;padding:.6rem 1rem;letter-spacing:.3px;border-top:1px solid rgba(255,255,255,.08);">
  Messenger Transfer Analytics &nbsp;·&nbsp; Built by <strong style="color:rgba(255,255,255,.7);">Hamzah Javaid</strong>
  &nbsp;·&nbsp; Hastings Direct &nbsp;·&nbsp; {max_date}
</div>

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
    # All queues that appear in transitions, ordered by case volume descending
    # so the highest-traffic queue is the default selection in QI + Journey dropdowns
    all_trans_queues = (
        df_raw["QUEUE_NEW"].dropna()
        .value_counts()
        .index.tolist()
    )

    print("\nGenerating index.html ...")
    html = generate_html(case_df, min_date, max_date, all_queues, all_trans_queues)
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
