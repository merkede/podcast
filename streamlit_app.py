# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(layout="wide", page_title="Case Routing Analytics")

# ==================================
# LOAD DATA
# ==================================
# Sample data for testing - replace with: df = pd.read_csv("data.csv")

@st.cache_data
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

# Generate sample data
df = generate_sample_data()

st.sidebar.info(f"📊 Using sample data: {df.CASE_ID.nunique()} cases, {len(df)} queue interactions")

# ==================================
# DATA PREP
# ==================================

@st.cache_data
def prepare_data(df):

    df = df.copy()

    # Ensure proper datetime parsing
    if 'CLOSE_DATETIME' in df.columns:
        df['CLOSE_DATETIME'] = pd.to_datetime(df['CLOSE_DATETIME'], errors='coerce')
    if 'CREATED_AT' in df.columns:
        df['CREATED_AT'] = pd.to_datetime(df['CREATED_AT'], errors='coerce')

    df = df.sort_values(["CASE_ID", "QUEUE_ORDER"])

    # case aggregates
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
        )
        .reset_index()
    )

    # loop flag
    loops = (
        df.groupby("CASE_ID")["QUEUE_NEW"]
        .apply(lambda x: x.duplicated().any())
        .astype(int)
        .reset_index(name="loop_flag")
    )

    case = case.merge(loops, on="CASE_ID", how="left")

    # Message intensity index (customer effort per agent minute)
    case['message_intensity'] = case['messages'] / (case['total_active_aht'] + 1)

    # Interaction density
    case['interaction_density'] = case['interactions'] / (case['total_active_aht'] + 1)

    # FTR flag
    case['ftr'] = (case['transfers'] == 0).astype(int)

    # Transfer bins for analysis
    case['transfer_bin'] = pd.cut(case['transfers'],
                                   bins=[-0.1, 0, 1, 2, 100],
                                   labels=['0', '1', '2', '3+'])

    return df, case


df, case = prepare_data(df)


# ==================================
# SIDEBAR FILTERS
# ==================================

st.sidebar.title("Filters")

queues = st.sidebar.multiselect(
    "Entry Queue",
    sorted(case.entry_queue.dropna().unique()),
    default=sorted(case.entry_queue.dropna().unique())
)

inh = st.sidebar.multiselect(
    "In Hours",
    [0, 1],
    default=[0, 1],
    format_func=lambda x: "Out of Hours" if x == 0 else "In Hours"
)

filtered = case[
    (case.entry_queue.isin(queues)) &
    (case.inhours.isin(inh))
].copy()


# ==================================
# TABS
# ==================================

tabs = st.tabs([
    "🎯 Executive Scorecard",
    "🔁 Process & Routing",
    "💰 Cost Inflation",
    "😤 Customer Friction",
    "⏰ In vs Out of Hours",
    "🔬 Queue Deep Dive",
    "📊 Transfer Flow Analysis",
    "📁 Case Explorer"
])


# ==================================
# TAB 1 — EXECUTIVE SCORECARD
# ==================================

with tabs[0]:

    st.title("Executive Scorecard")
    st.markdown("### North Star Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Cases", f"{len(filtered):,}")
    c2.metric("First Touch Resolution %",
              f"{(filtered.ftr.mean() * 100):.1f}%")
    c3.metric("Avg Transfers",
              f"{filtered.transfers.mean():.2f}")
    c4.metric("Median Routing Days",
              f"{filtered.routing_days.median():.1f}")
    c5.metric("Median Handle Time (min)",
              f"{filtered.total_active_aht.median():.0f}")

    st.markdown("---")
    st.markdown("### Process Drivers")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Loop/Rework Rate",
              f"{(filtered.loop_flag.mean() * 100):.1f}%")
    d2.metric("Median ASRT (min)",
              f"{filtered.asrt.median():.1f}")
    d3.metric("Avg Messages/Case",
              f"{filtered.messages.mean():.1f}")
    d4.metric("Multi-Transfer Cases",
              f"{((filtered.transfers >= 2).mean() * 100):.1f}%")

    st.markdown("---")
    st.markdown("### The Three Dimensions: Waiting vs Working vs Friction")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Waiting - time distribution
    data_wait = [
        filtered.routing_days.sum(),
        filtered.final_queue_days.sum()
    ]
    labels_wait = ["Routing\n(process waste)", "Final Queue\n(resolution)"]
    colors_wait = ['#e74c3c', '#3498db']
    axes[0].pie(data_wait, labels=labels_wait, autopct='%1.1f%%', colors=colors_wait, startangle=90)
    axes[0].set_title("WAITING: Calendar Days Distribution", fontweight='bold', fontsize=12)

    # Working - AHT by transfers
    transfer_aht = filtered.groupby('transfer_bin')['total_active_aht'].median().sort_index()
    axes[1].bar(range(len(transfer_aht)), transfer_aht.values, color='#2ecc71', alpha=0.7)
    axes[1].set_xticks(range(len(transfer_aht)))
    axes[1].set_xticklabels(transfer_aht.index)
    axes[1].set_xlabel("Number of Transfers", fontweight='bold')
    axes[1].set_ylabel("Median Handle Time (min)", fontweight='bold')
    axes[1].set_title("WORKING: AHT Inflation from Transfers", fontweight='bold', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    # Friction - message intensity
    msg_by_transfer = filtered.groupby('transfer_bin')['messages'].median().sort_index()
    axes[2].bar(range(len(msg_by_transfer)), msg_by_transfer.values, color='#e67e22', alpha=0.7)
    axes[2].set_xticks(range(len(msg_by_transfer)))
    axes[2].set_xticklabels(msg_by_transfer.index)
    axes[2].set_xlabel("Number of Transfers", fontweight='bold')
    axes[2].set_ylabel("Median Messages", fontweight='bold')
    axes[2].set_title("FRICTION: Customer Effort by Complexity", fontweight='bold', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Impact Heatmap: How Transfers Affect Everything")

    # Create correlation heatmap showing transfer impact
    impact_data = filtered.groupby('transfer_bin').agg({
        'routing_days': 'median',
        'total_active_aht': 'median',
        'asrt': 'median',
        'messages': 'median',
        'close_hours': 'median'
    }).T

    # Normalize to show % change from baseline (0 transfers)
    if '0' in impact_data.columns:
        baseline = impact_data['0']
        impact_pct = impact_data.div(baseline, axis=0) * 100 - 100
    else:
        impact_pct = impact_data

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(impact_pct, annot=True, fmt='.0f', cmap='RdYlGn_r',
                center=0, cbar_kws={'label': '% Change from FTR Baseline'},
                linewidths=1, linecolor='white', ax=ax)
    ax.set_xlabel("Number of Transfers", fontweight='bold', fontsize=11)
    ax.set_ylabel("Metric", fontweight='bold', fontsize=11)
    ax.set_yticklabels(['Routing Days', 'Handle Time', 'ASRT', 'Messages', 'Total Hours'], rotation=0)
    ax.set_title("Transfer Impact Matrix: % Increase vs First-Touch Resolution",
                 fontweight='bold', fontsize=13, pad=15)
    plt.tight_layout()
    st.pyplot(fig)


# ==================================
# TAB 2 — PROCESS & ROUTING
# ==================================

with tabs[1]:

    st.title("Process & Routing Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Bottleneck Queues: Pareto Analysis")

        # Queue-level delay contribution
        queue_impact = (
            df.groupby("QUEUE_NEW")
            .agg(
                total_delay_days=("DAYS_IN_QUEUE", "sum"),
                median_days=("DAYS_IN_QUEUE", "median"),
                p90_days=("DAYS_IN_QUEUE", lambda x: x.quantile(0.9)),
                volume=("CASE_ID", "nunique")
            )
            .sort_values("total_delay_days", ascending=False)
            .head(10)
            .reset_index()
        )

        queue_impact['cumulative_pct'] = (queue_impact['total_delay_days'].cumsum() /
                                          queue_impact['total_delay_days'].sum() * 100)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.barh(range(len(queue_impact)), queue_impact['total_delay_days'],
                 color='#e74c3c', alpha=0.7, label='Total Days')
        ax1.set_yticks(range(len(queue_impact)))
        ax1.set_yticklabels(queue_impact['QUEUE_NEW'])
        ax1.set_xlabel("Total Delay Days Contributed", fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        ax2 = ax1.twiny()
        ax2.plot(queue_impact['cumulative_pct'], range(len(queue_impact)),
                 'o-', color='#2c3e50', linewidth=2, markersize=6, label='Cumulative %')
        ax2.set_xlabel("Cumulative % of Total Delay", fontweight='bold', color='#2c3e50')
        ax2.tick_params(axis='x', labelcolor='#2c3e50')
        ax2.set_xlim(0, 105)

        plt.title("Top 10 Bottleneck Queues (80/20 Rule)", fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)

        # Format the dataframe
        display_df = queue_impact[['QUEUE_NEW', 'total_delay_days', 'median_days',
                                    'p90_days', 'volume', 'cumulative_pct']].copy()
        display_df['total_delay_days'] = display_df['total_delay_days'].round(0).astype(int)
        display_df['median_days'] = display_df['median_days'].round(1)
        display_df['p90_days'] = display_df['p90_days'].round(1)
        display_df['cumulative_pct'] = display_df['cumulative_pct'].round(1)

        st.dataframe(display_df, use_container_width=True)

    with col2:
        st.markdown("### Entry Queue Effectiveness")

        entry_performance = (
            filtered.groupby("entry_queue")
            .agg(
                cases=("CASE_ID", "count"),
                ftr_rate=("ftr", "mean"),
                avg_transfers=("transfers", "mean"),
                median_routing_days=("routing_days", "median"),
                median_aht=("total_active_aht", "median")
            )
            .sort_values("ftr_rate", ascending=True)
            .head(10)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create stacked bars
        bars1 = ax.barh(range(len(entry_performance)),
                        entry_performance['ftr_rate'] * 100,
                        color='#2ecc71', alpha=0.8, label='FTR %')
        bars2 = ax.barh(range(len(entry_performance)),
                        (1 - entry_performance['ftr_rate']) * 100,
                        left=entry_performance['ftr_rate'] * 100,
                        color='#e74c3c', alpha=0.8, label='Transfer %')

        ax.set_yticks(range(len(entry_performance)))
        ax.set_yticklabels(entry_performance['entry_queue'])
        ax.set_xlabel("Case Distribution (%)", fontweight='bold')
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)

        plt.title("Entry Queue FTR Performance (Worst to Best)", fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)

        # Format the dataframe
        display_entry = entry_performance.copy()
        display_entry['ftr_rate'] = (display_entry['ftr_rate'] * 100).round(1)
        display_entry['avg_transfers'] = display_entry['avg_transfers'].round(2)
        display_entry['median_routing_days'] = display_entry['median_routing_days'].round(1)
        display_entry['median_aht'] = display_entry['median_aht'].round(0).astype(int)
        display_entry = display_entry.rename(columns={'ftr_rate': 'FTR %'})

        st.dataframe(display_entry, use_container_width=True)

    st.markdown("---")
    st.markdown("### Routing Patterns: Loop Detection")

    loop_col1, loop_col2, loop_col3 = st.columns(3)

    loop_col1.metric("Cases with Rework",
                     f"{filtered.loop_flag.sum():,}",
                     f"{(filtered.loop_flag.mean() * 100):.1f}% of total")

    loop_cases = filtered[filtered.loop_flag == 1]
    if len(loop_cases) > 0:
        loop_col2.metric("Avg Days (Looped Cases)",
                        f"{loop_cases.routing_days.mean():.1f}",
                        f"+{(loop_cases.routing_days.mean() / filtered.routing_days.mean() - 1) * 100:.0f}% vs normal")
        loop_col3.metric("Avg AHT (Looped Cases)",
                        f"{loop_cases.total_active_aht.mean():.0f}",
                        f"+{(loop_cases.total_active_aht.mean() / filtered.total_active_aht.mean() - 1) * 100:.0f}% vs normal")


# ==================================
# TAB 3 — COST INFLATION
# ==================================

with tabs[2]:

    st.title("Cost Inflation Analysis")

    st.markdown("### The Cost of Complexity")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Handle Time Inflation per Transfer")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Violin plot for distribution
        parts = ax.violinplot([filtered[filtered.transfer_bin == cat]['total_active_aht'].dropna().values
                               for cat in ['0', '1', '2', '3+']],
                              positions=range(4), showmeans=True, showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.7)

        ax.set_xticks(range(4))
        ax.set_xticklabels(['0', '1', '2', '3+'])
        ax.set_xlabel("Number of Transfers", fontweight='bold')
        ax.set_ylabel("Total Active Handle Time (min)", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add median values as text
        medians = filtered.groupby('transfer_bin')['total_active_aht'].median()
        for i, cat in enumerate(['0', '1', '2', '3+']):
            if cat in medians.index:
                ax.text(i, medians[cat], f'{medians[cat]:.0f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.title("AHT Distribution by Transfer Count", fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Cost vs Waiting Decomposition")

        # Scatter with transfer count as color
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(filtered['routing_days'],
                            filtered['total_active_aht'],
                            c=filtered['transfers'],
                            cmap='YlOrRd',
                            alpha=0.6,
                            s=50,
                            edgecolors='black',
                            linewidth=0.5)

        ax.set_xlabel("Routing Days (Waiting)", fontweight='bold')
        ax.set_ylabel("Total Handle Time (Working)", fontweight='bold')
        ax.grid(alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Number of Transfers", fontweight='bold')

        plt.title("Waiting vs Working: Different Problems", fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Queue Cost Profile Heatmap")

    # Queue performance matrix
    queue_cost = (
        df.merge(case[['CASE_ID', 'transfers']], on='CASE_ID')
        .groupby(['QUEUE_NEW', 'transfers'])
        .agg(
            cases=('CASE_ID', 'nunique'),
            median_aht=('TOTALACTIVEAHT', 'median')
        )
        .reset_index()
        .pivot(index='QUEUE_NEW', columns='transfers', values='median_aht')
        .fillna(0)
    )

    # Limit to top queues by volume
    top_queues = df.groupby('QUEUE_NEW')['CASE_ID'].nunique().nlargest(15).index
    queue_cost_filtered = queue_cost.loc[queue_cost.index.isin(top_queues)]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(queue_cost_filtered, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Median AHT (min)'},
                linewidths=1, linecolor='white', ax=ax)
    ax.set_xlabel("Number of Transfers Before Queue", fontweight='bold')
    ax.set_ylabel("Queue", fontweight='bold')
    ax.set_title("Queue AHT by Arrival Complexity\n(Shows if queues are slow or receive damaged cases)",
                 fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig)


# ==================================
# TAB 4 — CUSTOMER FRICTION
# ==================================

with tabs[3]:

    st.title("Customer Friction & Experience")

    st.markdown("### Message Intensity: Customer Effort Signal")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Messages by Journey Complexity")

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(data=filtered, x='transfer_bin', y='messages',
                   palette='Reds', ax=ax)
        ax.set_xlabel("Number of Transfers", fontweight='bold')
        ax.set_ylabel("Messages Received from Customer", fontweight='bold')
        ax.set_title("Customer Effort Increases with Routing Friction", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Message Intensity Index")
        st.markdown("*Messages per minute of agent time*")

        fig, ax = plt.subplots(figsize=(10, 6))

        intensity_by_transfer = filtered.groupby('transfer_bin')['message_intensity'].median()
        bars = ax.bar(range(len(intensity_by_transfer)), intensity_by_transfer.values,
                     color='#e67e22', alpha=0.7)

        # Add value labels
        for i, v in enumerate(intensity_by_transfer.values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_xticks(range(len(intensity_by_transfer)))
        ax.set_xticklabels(intensity_by_transfer.index)
        ax.set_xlabel("Number of Transfers", fontweight='bold')
        ax.set_ylabel("Messages / AHT Minute", fontweight='bold')
        ax.set_title("Routing Complexity Creates Disproportionate Customer Effort", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### ASRT: Responsiveness Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ASRT vs Routing Days")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bins for routing days
        filtered_temp = filtered.copy()
        filtered_temp['routing_bin'] = pd.cut(filtered_temp['routing_days'],
                                              bins=[-0.1, 0, 2, 5, 100],
                                              labels=['0', '1-2', '3-5', '6+'])

        sns.boxplot(data=filtered_temp, x='routing_bin', y='asrt',
                   palette='Blues', ax=ax)
        ax.set_xlabel("Routing Days", fontweight='bold')
        ax.set_ylabel("ASRT (min)", fontweight='bold')
        ax.set_title("Conversational Responsiveness Degrades with Routing Complexity", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Experience Degradation Heatmap")

        # Create experience matrix: routing days vs transfers
        exp_matrix = filtered.copy()
        exp_matrix['routing_bin'] = pd.cut(exp_matrix['routing_days'],
                                           bins=[-0.1, 0, 2, 5, 100],
                                           labels=['0', '1-2', '3-5', '6+'])

        pivot_asrt = exp_matrix.groupby(['transfer_bin', 'routing_bin'])['asrt'].median().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_asrt, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Median ASRT (min)'},
                   linewidths=1, linecolor='white', ax=ax)
        ax.set_xlabel("Routing Days", fontweight='bold')
        ax.set_ylabel("Number of Transfers", fontweight='bold')
        ax.set_title("ASRT Degradation Matrix", fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)


# ==================================
# TAB 5 — IN VS OUT OF HOURS
# ==================================

with tabs[4]:

    st.title("In-Hours vs Out-of-Hours Analysis")

    st.markdown("### Structural Fairness Assessment")

    # Comparison metrics
    ih_data = filtered[filtered.inhours == 1]
    ooh_data = filtered[filtered.inhours == 0]

    if len(ooh_data) > 0:
        col1, col2, col3, col4 = st.columns(4)

        ih_ftr = ih_data.ftr.mean() * 100
        ooh_ftr = ooh_data.ftr.mean() * 100
        col1.metric("FTR Rate",
                   f"IH: {ih_ftr:.1f}%\nOOH: {ooh_ftr:.1f}%",
                   f"{ooh_ftr - ih_ftr:.1f}pp penalty")

        ih_routing = ih_data.routing_days.median()
        ooh_routing = ooh_data.routing_days.median()
        col2.metric("Median Routing Days",
                   f"IH: {ih_routing:.1f}\nOOH: {ooh_routing:.1f}",
                   f"+{((ooh_routing / ih_routing - 1) * 100):.0f}% slower" if ih_routing > 0 else "N/A")

        ih_aht = ih_data.total_active_aht.median()
        ooh_aht = ooh_data.total_active_aht.median()
        col3.metric("Median AHT",
                   f"IH: {ih_aht:.0f}\nOOH: {ooh_aht:.0f}",
                   f"+{((ooh_aht / ih_aht - 1) * 100):.0f}%" if ih_aht > 0 else "N/A")

        ih_asrt = ih_data.asrt.median()
        ooh_asrt = ooh_data.asrt.median()
        col4.metric("Median ASRT",
                   f"IH: {ih_asrt:.1f}\nOOH: {ooh_asrt:.1f}",
                   f"+{((ooh_asrt / ih_asrt - 1) * 100):.0f}%" if ih_asrt > 0 else "N/A")

        st.markdown("---")

        # Multi-metric comparison
        st.markdown("### Comprehensive Penalty View")

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

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Side-by-side bars
        x = np.arange(len(comparison_data))
        width = 0.35

        axes[0].barh(x - width/2, comparison_data['In Hours'], width,
                    label='In Hours', color='#2ecc71', alpha=0.8)
        axes[0].barh(x + width/2, comparison_data['Out of Hours'], width,
                    label='Out of Hours', color='#e74c3c', alpha=0.8)
        axes[0].set_yticks(x)
        axes[0].set_yticklabels(comparison_data['Metric'])
        axes[0].legend()
        axes[0].set_xlabel("Value", fontweight='bold')
        axes[0].set_title("Absolute Comparison", fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].invert_yaxis()

        # Penalty bars
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in comparison_data['% Difference']]
        axes[1].barh(comparison_data['Metric'], comparison_data['% Difference'],
                    color=colors, alpha=0.7)
        axes[1].axvline(x=0, color='black', linewidth=1)
        axes[1].set_xlabel("% Difference (OOH vs IH)", fontweight='bold')
        axes[1].set_title("Out-of-Hours Penalty", fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()

        # Add value labels
        for i, v in enumerate(comparison_data['% Difference']):
            axes[1].text(v, i, f'{v:+.0f}%', va='center',
                        ha='left' if v > 0 else 'right', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        # Format the dataframe
        display_comparison = comparison_data.copy()
        display_comparison['In Hours'] = display_comparison['In Hours'].round(1)
        display_comparison['Out of Hours'] = display_comparison['Out of Hours'].round(1)
        display_comparison['% Difference'] = display_comparison['% Difference'].round(1)

        st.dataframe(display_comparison, use_container_width=True)
    else:
        st.warning("No out-of-hours data available for comparison")


# ==================================
# TAB 6 — QUEUE DEEP DIVE
# ==================================

with tabs[5]:

    st.title("Queue Deep Dive")

    q_select = st.selectbox(
        "Select Queue for Analysis",
        sorted(df.QUEUE_NEW.dropna().unique())
    )

    subset_cases = df[df.QUEUE_NEW == q_select]["CASE_ID"].unique()
    subset = case[case.CASE_ID.isin(subset_cases)].copy()
    subset_df = df[df.QUEUE_NEW == q_select].copy()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Cases Touching Queue", f"{len(subset):,}")
    col2.metric("Median Days in Queue", f"{subset_df.DAYS_IN_QUEUE.median():.1f}")
    col3.metric("P90 Days in Queue", f"{subset_df.DAYS_IN_QUEUE.quantile(0.9):.1f}")
    col4.metric("Total Delay Contribution", f"{subset_df.DAYS_IN_QUEUE.sum():.0f} days")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Queue Dwell Distribution")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(subset_df.DAYS_IN_QUEUE, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(subset_df.DAYS_IN_QUEUE.median(), color='red',
                  linestyle='--', linewidth=2, label=f'Median: {subset_df.DAYS_IN_QUEUE.median():.1f}')
        ax.axvline(subset_df.DAYS_IN_QUEUE.quantile(0.9), color='orange',
                  linestyle='--', linewidth=2, label=f'P90: {subset_df.DAYS_IN_QUEUE.quantile(0.9):.1f}')
        ax.set_xlabel("Days in Queue", fontweight='bold')
        ax.set_ylabel("Number of Cases", fontweight='bold')
        ax.set_title(f"Dwell Time Distribution: {q_select}", fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("### Arrival Complexity Profile")
        st.markdown("*Does this queue receive 'damaged' cases?*")

        # Analyze arrival state
        arrival_analysis = (
            df[df.QUEUE_NEW == q_select]
            .merge(case[['CASE_ID', 'transfers']], on='CASE_ID')
            .groupby('QUEUE_ORDER')
            .agg(
                cases=('CASE_ID', 'nunique'),
                median_days=('DAYS_IN_QUEUE', 'median')
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax2 = ax.twinx()

        ax.bar(arrival_analysis['QUEUE_ORDER'], arrival_analysis['cases'],
              color='#95a5a6', alpha=0.5, label='Volume')
        ax2.plot(arrival_analysis['QUEUE_ORDER'], arrival_analysis['median_days'],
                'o-', color='#e74c3c', linewidth=2, markersize=8, label='Median Days')

        ax.set_xlabel("Queue Order (Transfer Count)", fontweight='bold')
        ax.set_ylabel("Number of Cases", fontweight='bold', color='#95a5a6')
        ax2.set_ylabel("Median Days in Queue", fontweight='bold', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')

        ax.set_title(f"How Late Do Cases Arrive at {q_select}?", fontweight='bold')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)


# ==================================
# TAB 7 — TRANSFER FLOW ANALYSIS
# ==================================

with tabs[6]:

    st.title("Transfer Flow Analysis")

    st.markdown("### Queue-to-Queue Transfer Patterns")

    # Create transfer pairs
    transfer_flows = []
    for case_id in df.CASE_ID.unique():
        case_journey = df[df.CASE_ID == case_id].sort_values('QUEUE_ORDER')
        queues = case_journey.QUEUE_NEW.tolist()
        for i in range(len(queues) - 1):
            transfer_flows.append({
                'from_queue': queues[i],
                'to_queue': queues[i + 1],
                'case_id': case_id
            })

    if transfer_flows:
        transfer_df = pd.DataFrame(transfer_flows)

        # Top transfer paths
        top_paths = (
            transfer_df.groupby(['from_queue', 'to_queue'])
            .size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
            .head(20)
        )

        st.markdown("### Top 20 Transfer Paths")

        fig, ax = plt.subplots(figsize=(12, 8))

        top_paths['path'] = top_paths['from_queue'] + ' → ' + top_paths['to_queue']

        bars = ax.barh(range(len(top_paths)), top_paths['count'], color='#3498db', alpha=0.7)
        ax.set_yticks(range(len(top_paths)))
        ax.set_yticklabels(top_paths['path'])
        ax.set_xlabel("Number of Transfers", fontweight='bold')
        ax.set_title("Most Common Transfer Paths", fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(top_paths['count']):
            ax.text(v, i, f' {v}', va='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Transfer Matrix Heatmap")

        # Create transfer matrix
        transfer_matrix = transfer_df.groupby(['from_queue', 'to_queue']).size().unstack(fill_value=0)

        # Limit to top queues
        top_queues_by_transfers = (
            transfer_df.groupby('from_queue').size().nlargest(15).index.tolist()
        )

        transfer_matrix_filtered = transfer_matrix.loc[
            transfer_matrix.index.isin(top_queues_by_transfers),
            transfer_matrix.columns.isin(top_queues_by_transfers)
        ]

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(transfer_matrix_filtered, annot=True, fmt='d', cmap='YlOrRd',
                   cbar_kws={'label': 'Transfer Count'},
                   linewidths=0.5, linecolor='white', ax=ax,
                   square=True)
        ax.set_xlabel("To Queue", fontweight='bold', fontsize=12)
        ax.set_ylabel("From Queue", fontweight='bold', fontsize=12)
        ax.set_title("Transfer Flow Matrix\n(Diagonal = loops, Off-diagonal = handoffs)",
                    fontweight='bold', fontsize=14, pad=20)

        # Highlight diagonal (loops)
        for i in range(min(len(transfer_matrix_filtered), len(transfer_matrix_filtered.columns))):
            if transfer_matrix_filtered.index[i] in transfer_matrix_filtered.columns:
                col_idx = transfer_matrix_filtered.columns.get_loc(transfer_matrix_filtered.index[i])
                ax.add_patch(Rectangle((col_idx, i), 1, 1, fill=False,
                                      edgecolor='blue', lw=3))

        plt.tight_layout()
        st.pyplot(fig)

        # Loop detection
        st.markdown("---")
        st.markdown("### Loop/Rework Patterns")

        # Find actual loops
        loop_patterns = transfer_df[transfer_df.from_queue == transfer_df.to_queue]

        if len(loop_patterns) > 0:
            loop_summary = loop_patterns.groupby('from_queue').size().reset_index(name='loop_count')
            loop_summary = loop_summary.sort_values('loop_count', ascending=False)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(loop_summary, use_container_width=True)

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(loop_summary)), loop_summary['loop_count'],
                      color='#e74c3c', alpha=0.7)
                ax.set_xticks(range(len(loop_summary)))
                ax.set_xticklabels(loop_summary['from_queue'], rotation=45, ha='right')
                ax.set_ylabel("Number of Self-Loops", fontweight='bold')
                ax.set_title("Queues with Rework (Same Queue Repeated)", fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No direct self-loops detected in the data")
    else:
        st.warning("No transfer data available")


# ==================================
# TAB 8 — CASE EXPLORER
# ==================================

with tabs[7]:

    st.title("Case Explorer")

    cid = st.selectbox("Select case ID", filtered.CASE_ID.unique())

    journey = df[df.CASE_ID == cid].sort_values("QUEUE_ORDER")
    case_summary = case[case.CASE_ID == cid].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Transfers", int(case_summary['transfers']))
    col2.metric("Routing Days", f"{case_summary['routing_days']:.1f}")
    col3.metric("Total AHT", f"{case_summary['total_active_aht']:.0f}")
    col4.metric("Messages", int(case_summary['messages']))
    col5.metric("ASRT", f"{case_summary['asrt']:.1f}")

    st.markdown("### Journey Path")

    # Visualize journey
    fig, ax = plt.subplots(figsize=(12, 4))

    for i, row in journey.iterrows():
        order = row['QUEUE_ORDER']
        queue = row['QUEUE_NEW']
        days = row['DAYS_IN_QUEUE']

        # Draw box
        ax.add_patch(Rectangle((order - 0.4, 0), 0.8, 1,
                              facecolor='#3498db' if order < len(journey) else '#2ecc71',
                              edgecolor='black', linewidth=2))

        # Queue name
        ax.text(order, 0.7, queue, ha='center', va='center',
               fontweight='bold', fontsize=10, wrap=True)

        # Days
        ax.text(order, 0.3, f"{days:.1f} days", ha='center', va='center',
               fontsize=9, style='italic')

        # Arrow to next
        if order < len(journey):
            ax.arrow(order + 0.4, 0.5, 0.15, 0, head_width=0.2,
                    head_length=0.05, fc='black', ec='black')

    ax.set_xlim(0.5, len(journey) + 0.5)
    ax.set_ylim(-0.2, 1.3)
    ax.axis('off')
    ax.set_title(f"Case {cid} Journey Timeline", fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Detailed Journey Data")
    st.dataframe(journey, use_container_width=True)
