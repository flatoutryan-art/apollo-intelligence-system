import os
import pandas as pd
import anthropic
import plotly.express as px
import plotly.graph_objects as go
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 1. Configuration & Path Setup
# Using your specific absolute path to ensure no ambiguity
base_dir = r"C:\Users\RyanVanEs\OneDrive - apolloafrica.co.za 1\Desktop\Projects\Apollo_Intelligence_System"
EXPORT_DIR = os.path.join(base_dir, "export")
os.makedirs(EXPORT_DIR, exist_ok=True)

DATA_PATH = os.path.join(base_dir, "knowledge", "market_data.csv")
SCRAPED_DATA_PATH = r"C:\Users\RyanVanEs\OneDrive - apolloafrica.co.za 1\Desktop\Projects\Apollo_Intelligence_System\export\market_data_cleaned.json"
REPORT_PATH = os.path.join(EXPORT_DIR, "market_report.md")
JSON_PATH = os.path.join(EXPORT_DIR, "market_data_cleaned.json")
DASHBOARD_PATH = os.path.join(EXPORT_DIR, "dashboard.html")
MODEL_ID = "claude-haiku-4-5-20251001"
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

print(f"Working directory: {base_dir}")
print(f"Export directory: {EXPORT_DIR}")
print(f"ANTHROPIC_API_KEY found: {bool(API_KEY)}")


def load_scraped_data():
    """Load validated scraped data source for merge."""
    if not os.path.exists(SCRAPED_DATA_PATH):
        return pd.DataFrame(), None
    if SCRAPED_DATA_PATH.lower().endswith(".json"):
        return pd.read_json(SCRAPED_DATA_PATH), SCRAPED_DATA_PATH
    if SCRAPED_DATA_PATH.lower().endswith(".csv"):
        return pd.read_csv(SCRAPED_DATA_PATH, header=6, sep=None, engine='python', encoding='latin-1', on_bad_lines='skip'), SCRAPED_DATA_PATH
    return pd.DataFrame(), None

# 2. Robust Data Loading
try:
    df = pd.read_csv(DATA_PATH, header=6, sep=None, engine='python', encoding='latin-1', on_bad_lines='skip')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Build join key in base dataset.
    if 'Competitors Tracked' in df.columns:
        df['Competitor'] = df['Competitors Tracked'].astype(str).str.strip()
    elif 'Competitor' not in df.columns:
        df['Competitor'] = 'Unknown'

    # Load scraped dataset and normalize join key.
    df_scraped, scraped_source_path = load_scraped_data()
    if not df_scraped.empty:
        if 'IPP_Name' in df_scraped.columns:
            df_scraped['Competitor'] = df_scraped['IPP_Name'].astype(str).str.strip()
        elif 'Competitor' in df_scraped.columns:
            df_scraped['Competitor'] = df_scraped['Competitor'].astype(str).str.strip()
        else:
            first_col = df_scraped.columns[0]
            df_scraped['Competitor'] = df_scraped[first_col].astype(str).str.strip()

        keep_cols = ['Competitor', 'IPP_Name', 'Energy_Produced', 'PPA_Signed', 'PPA_Signed_Value', 'Trading_License_Status', 'Trading_License']
        available_keep_cols = [c for c in keep_cols if c in df_scraped.columns]
        df_scraped = df_scraped[available_keep_cols].copy()
        df = df.merge(df_scraped, on='Competitor', how='left')
        print(f"Scraped source loaded: {scraped_source_path}")
    else:
        print("Scraped source not found. Using base dataset only.")

    # Robust null handling requested by user.
    # Coerce expected numeric fields before type-based fill to ensure they receive 0 values.
    numeric_targets = ['Energy_Produced', 'PPA_Signed_Value', 'PPA_Signed', 'Deals Logged', 'Regulatory Updates']
    for col in numeric_targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('Unknown'))

    # Save cleaned data
    df.to_json(JSON_PATH, orient='records', indent=4)
    audit_df = pd.read_json(JSON_PATH)
    print('--- DATA AUDIT ---')
    print('Available columns:', audit_df.columns.tolist())
    print(audit_df.head(5))
    print('------------------')
    competitive_intel_columns = [
        'Trading_License_Status',
        'I_REC_Volume',
        'Wheeling_Capacity',
        'Eskom_Direct_Feasibility'
    ]
    available_intel_columns = [col for col in competitive_intel_columns if col in df.columns]
    if available_intel_columns:
        market_context = df[available_intel_columns].head(50).to_string(index=False)
    else:
        market_context = df.head(50).to_string(index=False)
    print(f"--- Data Cleaned and Ready ---")
except Exception as e:
    print(f"CRITICAL DATA ERROR: {e}")
    exit()

# 3. Initialize Client
if not API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. Please export it in your environment before running agents.py."
    )

client = anthropic.Anthropic(api_key=API_KEY)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=12),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def generate_ai_text(prompt, max_tokens=1000):
    message = client.messages.create(
        model=MODEL_ID,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def generate_dashboard(strategic_strike_memo):
    dashboard_df = pd.read_json(JSON_PATH)

    if dashboard_df.empty:
        raise ValueError("Cleaned market dataset is empty; cannot build dashboard.")

    # Explicitly map scraped fields into dashboard model.
    scraped_mappings = {
        'Competitor': ['Competitor', 'IPP_Name', 'Competitors Tracked'],
        'Energy_Produced': ['Energy_Produced'],
        'PPA_Signed': ['PPA_Signed', 'PPA_Signed_Value'],
        'Trading_License_Status': ['Trading_License_Status', 'Trading_License', 'Regulatory Updates'],
    }
    for target_col, aliases in scraped_mappings.items():
        source_col = next((candidate for candidate in aliases if candidate in dashboard_df.columns), None)
        if source_col:
            dashboard_df[target_col] = dashboard_df[source_col]
        elif target_col in ['Energy_Produced', 'PPA_Signed']:
            dashboard_df[target_col] = 0
        else:
            dashboard_df[target_col] = "Unknown"

    dashboard_df['Energy_Produced'] = pd.to_numeric(dashboard_df['Energy_Produced'], errors='coerce').fillna(0)
    dashboard_df['PPA_Signed'] = pd.to_numeric(dashboard_df['PPA_Signed'], errors='coerce').fillna(0)
    dashboard_df['Trading_License_Status'] = dashboard_df['Trading_License_Status'].fillna("Unknown")

    # Competitive Intel dictionary (target schema requested for KPI logic)
    competitive_intel_columns = [
        'Trading_License_Status',
        'I_REC_Volume',
        'Wheeling_Capacity',
        'Eskom_Direct_Feasibility'
    ]

    # Resolve column mapping from available source columns.
    # If expected columns are missing, create safe defaults so dashboard generation never breaks.
    if 'Competitor' in dashboard_df.columns:
        dashboard_df['Competitor'] = dashboard_df['Competitor']
    elif 'Competitors Tracked' in dashboard_df.columns:
        dashboard_df['Competitor'] = dashboard_df['Competitors Tracked']
    elif 'Competitor' not in dashboard_df.columns:
        dashboard_df['Competitor'] = 'Unknown Competitor'

    column_aliases = {
        'Trading_License_Status': ['Trading_License_Status', 'Regulatory Updates'],
        'I_REC_Volume': ['I_REC_Volume'],
        'Wheeling_Capacity': ['Wheeling_Capacity', 'Deals Logged'],
        'Eskom_Direct_Feasibility': ['Eskom_Direct_Feasibility'],
    }

    for target_col, aliases in column_aliases.items():
        source_col = next((candidate for candidate in aliases if candidate in dashboard_df.columns), None)
        if source_col:
            dashboard_df[target_col] = dashboard_df[source_col]
        else:
            dashboard_df[target_col] = "N/A"

    # Use the full merged dataset (no name-based locking), with only basic null/blank cleanup.
    competitor_df = dashboard_df[['Competitor'] + competitive_intel_columns].copy()
    competitor_df = competitor_df[competitor_df['Competitor'].notna()]
    competitor_df['Competitor'] = competitor_df['Competitor'].astype(str).str.strip()
    competitor_df = competitor_df[competitor_df['Competitor'] != ""]

    competitor_df['Wheeling_Capacity_Numeric'] = pd.to_numeric(competitor_df['Wheeling_Capacity'], errors='coerce').fillna(0.0)
    competitor_df['Energy_Produced'] = pd.to_numeric(dashboard_df.get('Energy_Produced'), errors='coerce').fillna(0.0)
    competitor_df['PPA_Signed'] = pd.to_numeric(dashboard_df.get('PPA_Signed'), errors='coerce').fillna(0.0)

    # Derive numeric regulatory risk score from available data.
    # If a numeric field exists, use it; otherwise map textual license risk indicators.
    raw_reg = competitor_df['Trading_License_Status'].astype(str)
    mapped_risk = (
        raw_reg.str.lower()
        .map(
            lambda v: 2.0 if any(k in v for k in ['active', 'granted', 'green', 'yes'])
            else (8.0 if any(k in v for k in ['challenge', 'suspend', 'unknown', 'no', 'red']) else 5.0)
        )
    )
    numeric_reg = pd.to_numeric(competitor_df['Trading_License_Status'], errors='coerce')
    competitor_df['Regulatory Risk'] = numeric_reg.fillna(mapped_risk)

    # Normalize plot columns to requested naming.
    competitor_df['Wheeling Capacity'] = competitor_df['Wheeling_Capacity_Numeric']

    # Market headroom model (ceiling defaults if no explicit capacity in source).
    total_grid_capacity = float(
        pd.to_numeric(dashboard_df.get('Total Grid Capacity'), errors='coerce').fillna(0).max()
    ) if 'Total Grid Capacity' in dashboard_df.columns else 1000.0
    cumulative_competitor_capacity = float(competitor_df['Wheeling_Capacity_Numeric'].sum())
    available_grid_headroom = max(total_grid_capacity - cumulative_competitor_capacity, 0.0)

    # Actionable KPI calculations
    top3_capacity = float(competitor_df.nlargest(3, 'Wheeling_Capacity_Numeric')['Wheeling_Capacity_Numeric'].sum())
    market_concentration = (top3_capacity / cumulative_competitor_capacity * 100.0) if cumulative_competitor_capacity > 0 else 0.0

    # Average time-to-license fallback strategy:
    # 1) direct numeric column if present
    # 2) parse duration-like values from any matching column names
    # 3) fallback to 0
    avg_time_to_license = 0.0
    time_col_candidates = [c for c in dashboard_df.columns if 'license' in c.lower() and ('time' in c.lower() or 'duration' in c.lower())]
    if time_col_candidates:
        durations = pd.to_numeric(dashboard_df[time_col_candidates[0]], errors='coerce').fillna(0)
        avg_time_to_license = float(durations.mean()) if not durations.empty else 0.0
    # Column remap to business headers required by ExCo.
    competitor_df = competitor_df.rename(columns={
        'Eskom_Direct_Feasibility': 'Eskom_Feasibility'
    })
    display_columns = ['Competitor', 'Trading_License_Status', 'I_REC_Volume', 'Wheeling_Capacity', 'Eskom_Feasibility']
    top_competitors_df = (
        competitor_df
        .sort_values('Wheeling_Capacity_Numeric', ascending=False)
        .head(20)
        [display_columns]
        .copy()
    )

    # Apollo vs Market calculations
    market_total_wheeling_capacity = float(competitor_df['Wheeling_Capacity_Numeric'].sum())
    apollo_capacity = float(
        competitor_df.loc[
            competitor_df['Competitor'].str.contains("apollo", case=False, na=False),
            'Wheeling_Capacity_Numeric'
        ].sum()
    )
    active_license_count = int(
        competitor_df['Trading_License_Status']
        .astype(str)
        .str.contains("active|granted|yes|green", case=False, na=False)
        .sum()
    )

    # Explicit KPI table rendering: map required business columns directly
    kpi_table_rows = []
    for _, row in top_competitors_df.iterrows():
        kpi_table_rows.append(
            f"""
            <tr>
                <td>{row['Competitor']}</td>
                <td>{row['Trading_License_Status']}</td>
                <td>{row['I_REC_Volume']}</td>
                <td>{row['Wheeling_Capacity']}</td>
                <td>{row['Eskom_Feasibility']}</td>
            </tr>
            """
        )
    kpi_table_html = f"""
        <table class="table table-striped table-hover table-sm align-middle mb-0">
            <thead class="table-light">
                <tr>
                    <th>Competitor</th>
                    <th>Trading License Status</th>
                    <th>I-REC Volume</th>
                    <th>Wheeling Capacity</th>
                    <th>Eskom Feasibility</th>
                </tr>
            </thead>
            <tbody>
                {''.join(kpi_table_rows) if kpi_table_rows else '<tr><td colspan="5">No competitor KPI rows available.</td></tr>'}
            </tbody>
        </table>
    """

    kpi_summary_html = f"""
        <div class="row g-3 mb-3">
            <div class="col-md-4">
                <div class="p-3 bg-light rounded border border-2">
                    <div class="small text-muted">Market Concentration (Top 3)</div>
                    <div class="h5 mb-0">{market_concentration:.1f}%</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3 bg-light rounded border border-2">
                    <div class="small text-muted">Average Time-to-License</div>
                    <div class="h5 mb-0">{avg_time_to_license:.1f} months</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3 bg-light rounded border border-2">
                    <div class="small text-muted">Available Grid Headroom</div>
                    <div class="h5 mb-0">{available_grid_headroom:,.1f}</div>
                </div>
            </div>
        </div>
    """

    report_text = ""
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as report_file:
            report_text = report_file.read()

    chart_html = """
        <div class="alert alert-secondary mb-0">
            Chart is being generated. If it fails, KPI table and memos remain available.
        </div>
    """
    headroom_html = chart_html
    energy_chart_html = chart_html
    ppa_chart_html = chart_html
    license_chart_html = chart_html

    def build_html(scatter_block, headroom_block, energy_block, ppa_block, license_block):
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Apollo Intelligence Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #004B49; color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }}
        .card {{ background-color: #006B68; border: 1px solid #A4D65E; color: #FFFFFF; border-radius: 0.75rem; }}
        .table {{ color: #FFFFFF; }}
        th {{ background-color: #A4D65E; color: #004B49; }}
        .card-title {{ margin-bottom: 0.75rem; color: #A4D65E; }}
        .memo-text {{ white-space: pre-wrap; }}
        .small-muted {{ color: #E2E8F0; font-size: 0.9rem; }}
        .brand-title {{ color: #A4D65E; }}
    </style>
</head>
<body>
    <div class="container py-4">
        <div class="mb-4">
            <h1 class="h3 mb-1 brand-title">Apollo Intelligence Dashboard</h1>
            <p class="small-muted mb-0">Competitive intelligence snapshot generated from cleaned market data and AI synthesis.</p>
        </div>
        <div class="row g-4">
            <div class="col-lg-7">
                <div class="card shadow-sm">
                    <div class="card-body">
                        <h2 class="h5 card-title">Strategic Quadrant: Wheeling Capacity vs Regulatory Risk</h2>
                        {scatter_block}
                    </div>
                </div>
            </div>
            <div class="col-lg-5">
                <div class="card shadow-sm h-100">
                    <div class="card-body">
                        <h2 class="h5 card-title">Market Headroom</h2>
                        {headroom_block}
                    </div>
                </div>
            </div>
            <div class="col-lg-7">
                <div class="card shadow-sm h-100">
                    <div class="card-body">
                        <h2 class="h5 card-title">KPI Table</h2>
                        {kpi_summary_html}
                        <div class="table-responsive">
                            {kpi_table_html}
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-5">
                <div class="card shadow-sm h-100">
                    <div class="card-body">
                        <h2 class="h5 card-title">Strategic Strike Memo</h2>
                        <div class="memo-text">{strategic_strike_memo}</div>
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="card shadow-sm h-100">
                    <div class="card-body">
                        <h2 class="h6 card-title">Energy Produced</h2>
                        {energy_block}
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="card shadow-sm h-100">
                    <div class="card-body">
                        <h2 class="h6 card-title">PPA Signed</h2>
                        {ppa_block}
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="card shadow-sm h-100">
                    <div class="card-body">
                        <h2 class="h6 card-title">Trading License Status</h2>
                        {license_block}
                    </div>
                </div>
            </div>
            <div class="col-12">
                <div class="card shadow-sm">
                    <div class="card-body">
                        <h2 class="h6 card-title">Full Market Report</h2>
                        <div class="memo-text">{report_text}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    # Always create dashboard file before chart attempt.
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as dashboard_file:
        dashboard_file.write(build_html(chart_html, headroom_html, energy_chart_html, ppa_chart_html, license_chart_html))

    try:
        chart_df = competitor_df.copy()
        chart_df['Energy_Produced'] = pd.to_numeric(chart_df['Energy_Produced'], errors='coerce')
        chart_df['PPA_Signed'] = pd.to_numeric(chart_df['PPA_Signed'], errors='coerce')
        chart_df[['Energy_Produced', 'PPA_Signed']] = chart_df[['Energy_Produced', 'PPA_Signed']].fillna(0)
        chart_df = chart_df[(chart_df['Energy_Produced'] > 0) | (chart_df['PPA_Signed'] > 0)]

        if chart_df.empty:
            raise ValueError("No valid numeric rows for scatter plot.")

        x_median = chart_df['Energy_Produced'].median()
        y_median = chart_df['PPA_Signed'].median()

        hover_name = 'Competitor' if 'Competitor' in chart_df.columns else None
        scatter_fig = px.scatter(
            chart_df,
            x='Energy_Produced',
            y='PPA_Signed',
            hover_name=hover_name,
            title="Energy Produced vs PPA Signed",
            color='Energy_Produced',
            color_continuous_scale='Blues',
        )
        scatter_fig.add_vline(x=x_median, line_dash='dash', line_color='#0b1f3a')
        scatter_fig.add_hline(y=y_median, line_dash='dash', line_color='#0b1f3a')
        scatter_fig.add_annotation(x=x_median * 1.35 if x_median else 1, y=y_median * 1.35 if y_median else 1, text="Dominant Players", showarrow=False, font=dict(color='#0b1f3a'))
        scatter_fig.add_annotation(x=x_median * 1.35 if x_median else 1, y=max(y_median * 0.65, 0.2), text="Emerging Threats", showarrow=False, font=dict(color='#0b1f3a'))
        scatter_fig.add_annotation(x=max(x_median * 0.65, 0.2), y=y_median * 1.35 if y_median else 1, text="Niche Players", showarrow=False, font=dict(color='#0b1f3a'))
        scatter_fig.add_annotation(x=max(x_median * 0.65, 0.2), y=max(y_median * 0.65, 0.2), text="Market Noise", showarrow=False, font=dict(color='#0b1f3a'))
        scatter_fig.update_layout(template='plotly_white', height=480)
        chart_html = scatter_fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Market headroom waterfall:
        # Start from total grid capacity, subtract cumulative competitor capacity, show remaining headroom.
        headroom_fig = go.Figure(
            go.Waterfall(
                name="Headroom Flow",
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["Total Grid Capacity", "Cumulative Competitor Capacity", "Grid Headroom"],
                y=[total_grid_capacity, -cumulative_competitor_capacity, 0],
                connector={"line": {"color": "#0b1f3a"}},
                decreasing={"marker": {"color": "#0f3d7a"}},
                increasing={"marker": {"color": "#60a5fa"}},
                totals={"marker": {"color": "#f5a623"}},
                text=[
                    f"{total_grid_capacity:,.1f}",
                    f"-{cumulative_competitor_capacity:,.1f}",
                    f"{available_grid_headroom:,.1f}",
                ],
                textposition="outside",
            )
        )
        headroom_fig.update_layout(
            template='plotly_white',
            height=480,
            showlegend=False,
            yaxis_title='Capacity',
        )
        headroom_html = headroom_fig.to_html(full_html=False, include_plotlyjs=False)

        # Separate chart block: Energy Produced
        energy_df = chart_df[['Competitor', 'Energy_Produced']].copy()
        energy_df = energy_df.sort_values('Energy_Produced', ascending=False).head(10)
        energy_fig = px.bar(energy_df, x='Competitor', y='Energy_Produced', title='Top Competitors by Energy Produced', color='Energy_Produced', color_continuous_scale='Blues')
        energy_fig.update_layout(template='plotly_white', height=360, xaxis_tickangle=-35)
        energy_chart_html = energy_fig.to_html(full_html=False, include_plotlyjs=False)

        # Separate chart block: PPA Signed
        ppa_df = chart_df[['Competitor', 'PPA_Signed']].copy()
        ppa_df = ppa_df.sort_values('PPA_Signed', ascending=False).head(10)
        ppa_fig = px.bar(ppa_df, x='Competitor', y='PPA_Signed', title='Top Competitors by PPA Signed', color='PPA_Signed', color_continuous_scale='Blues')
        ppa_fig.update_layout(template='plotly_white', height=360, xaxis_tickangle=-35)
        ppa_chart_html = ppa_fig.to_html(full_html=False, include_plotlyjs=False)

        # Separate chart block: Trading License Status
        license_df = competitor_df[['Trading_License_Status']].copy()
        license_df['Trading_License_Status'] = license_df['Trading_License_Status'].astype(str).str.strip()
        license_df = license_df[license_df['Trading_License_Status'] != ""]
        license_count_df = license_df.value_counts().reset_index(name='Count').head(10)
        license_fig = px.bar(
            license_count_df,
            x='Trading_License_Status',
            y='Count',
            title='Trading License Status Distribution',
            color='Count',
            color_continuous_scale='Blues'
        )
        license_fig.update_layout(template='plotly_white', height=360, xaxis_tickangle=-25)
        license_chart_html = license_fig.to_html(full_html=False, include_plotlyjs=False)
    except Exception as chart_error:
        chart_html = (
            "<div class=\"alert alert-warning mb-0\">"
            f"Chart generation failed: {chart_error}"
            "</div>"
        )
        headroom_html = chart_html
        energy_chart_html = chart_html
        ppa_chart_html = chart_html
        license_chart_html = chart_html

    with open(DASHBOARD_PATH, "w", encoding="utf-8") as dashboard_file:
        dashboard_file.write(build_html(chart_html, headroom_html, energy_chart_html, ppa_chart_html, license_chart_html))


# 4. Generate AI Market Report and Dashboard
try:
    report_text = generate_ai_text(
        f"Analyze this intelligence data for Apollo Africa. Provide 3 key observations:\n\n{market_context}",
        max_tokens=1000
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    strategic_strike_memo = generate_ai_text(
        (
            "Using this market dataset, write a concise 'Strategic Strike' memo for Apollo Africa. "
            "Identify high-potential municipal nodes for aggressive contracting, justify choices using "
            "Market Velocity and Regulatory Fragility, and include 3 action recommendations.\n\n"
            f"{market_context}"
        ),
        max_tokens=700
    )

    # Verification: confirm scraped numeric business fields exist before dashboard generation.
    if os.path.exists(JSON_PATH):
        verify_df = pd.read_json(JSON_PATH)
        verify_df['Energy_Produced'] = pd.to_numeric(verify_df['Energy_Produced'], errors='coerce').fillna(0) if 'Energy_Produced' in verify_df.columns else 0
        verify_df['PPA_Signed'] = pd.to_numeric(verify_df['PPA_Signed'], errors='coerce').fillna(0) if 'PPA_Signed' in verify_df.columns else 0
        print(f"Total merged records: {len(verify_df)}")
        print(f"Sample of merged data: {verify_df.head()}")
        print(f"Non-zero Energy_Produced rows: {(verify_df['Energy_Produced'] > 0).sum()}")
        print(f"Non-zero PPA_Signed rows: {(verify_df['PPA_Signed'] > 0).sum()}")

    generate_dashboard(strategic_strike_memo)

    print(f"\n[SUCCESS] Files created in: {EXPORT_DIR}")
    print(f"- Report: {REPORT_PATH}")
    print(f"- Data: {JSON_PATH}")
    print(f"- Dashboard: {DASHBOARD_PATH}")

except Exception as e:
    print(f"ERROR: {e}")