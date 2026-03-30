import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="WaterOps AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }

    /* ── Pipeline cards ── */
    .pipeline-card {
        background-color: #1E2127; border-radius: 8px;
        padding: 15px; text-align: center; border-bottom: 3px solid #E0E0E0;
    }

    /* ── FIX 1 & 2: Lighten ALL st.metric labels and values ── */
    [data-testid="stMetricLabel"] p {
        color: #A0AEC0 !important;   /* soft grey-blue label */
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #E2E8F0 !important;   /* near-white value */
    }

    /* ── FIX 1: Color the Network Balance delta badges ── */
    /* Surplus  → bright green */
    [data-testid="stMetricDelta"][data-direction="good"] svg,
    [data-testid="stMetricDelta"][data-direction="good"] span {
        color: #48BB78 !important;
    }
    /* Deficit  → bright red */
    [data-testid="stMetricDelta"][data-direction="bad"]  svg,
    [data-testid="stMetricDelta"][data-direction="bad"]  span {
        color: #FC8181 !important;
    }

    /* ── FIX 1: Explicit colored metric values via custom classes ── */
    .metric-green [data-testid="stMetricValue"] { color: #68D391 !important; }
    .metric-red   [data-testid="stMetricValue"] { color: #FC8181 !important; }
    .metric-white [data-testid="stMetricValue"] { color: #E2E8F0 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADERS (GIS & REAL CSVs)
# ==========================================
DATA_DIR = "map_data"

@st.cache_data
def load_and_fix_data(filepath):
    if os.path.exists(filepath):
        try:
            gdf = gpd.read_file(filepath)
            if gdf.crs and gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            elif not gdf.crs:
                gdf.set_crs("EPSG:27700", allow_override=True)
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except Exception:
            return None
    return None

@st.cache_data
def load_real_or_mock_demand(zone):
    csv_path = f"{DATA_DIR}/demand_data_day_wise.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        zone_df = df[df['Name'] == zone]
        if not zone_df.empty:
            date_cols = [c for c in df.columns if c not in ['Name', 'Avg_demand']]
            melted = zone_df.melt(id_vars=['Name'], value_vars=date_cols, var_name='Date_Str', value_name='Demand')
            melted['Demand'] = melted['Demand'].astype(str).str.replace('"', '').str.replace(',', '').astype(float)
            current_year = datetime.now().year
            melted['Date'] = pd.to_datetime(melted['Date_Str'] + f" {current_year}", format='%b_%d %Y', errors='coerce')
            df_fcst = melted.sort_values('Date').dropna(subset=['Date'])

            start_date = df_fcst['Date'].min()
            dates_hist = [start_date - timedelta(days=x) for x in range(14, 0, -1)]
            base = df_fcst['Demand'].iloc[0]
            hist = [base + np.random.normal(0, base * 0.02) for _ in range(14)]
            df_hist = pd.DataFrame({'Date': dates_hist, 'Demand': hist})

            metrics = {"Champion Model": "Hybrid Prophet + SARIMAX", "MAE": "Deployed", "Seasonality": "Calculated"}
            return df_hist, df_fcst, metrics, start_date

    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    dates_hist = [today - timedelta(days=x) for x in range(14, 0, -1)]
    base = 60 if zone == "Zone 5" else np.random.randint(40, 80)
    hist = [base + np.random.normal(0, 5) for _ in range(14)]
    dates_fcst = [today + timedelta(days=x) for x in range(0, 7)]
    fcst = [hist[-1] + np.random.normal(0, 3) + (x * 0.5) for x in range(7)]
    df_hist = pd.DataFrame({'Date': dates_hist, 'Demand': hist})
    df_fcst = pd.DataFrame({'Date': dates_fcst, 'Demand': fcst, 'Upper': [v + 4 for v in fcst], 'Lower': [v - 4 for v in fcst]})
    metrics = {"Champion Model": "Mock Data Active", "MAE": 1.24, "Seasonality": "High"}
    return df_hist, df_fcst, metrics, today

@st.cache_data
def load_optimization_plan():
    csv_path = f"{DATA_DIR}/optimized_schedule.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

pipes_gdf  = load_and_fix_data(f"{DATA_DIR}/Optimized_Pipes_3D.geojson")
prod_gdf   = load_and_fix_data(f"{DATA_DIR}/water_prod_nodes.geojson")
dsr_gdf    = load_and_fix_data(f"{DATA_DIR}/dsr_master_list.geojson")
demand_gdf = load_and_fix_data(f"{DATA_DIR}/water_demand_nodes.geojson")
opt_plan_df = load_optimization_plan()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("💧 WaterOps AI")
    st.markdown("---")
    page = st.radio("Navigation", ["Demand Forecasting", "Supply Optimization (Map)"])
    st.markdown("---")
    selected_zone = st.selectbox("Global Region Filter", [f"Zone {i}" for i in range(5, 11)])

# ==========================================
# 4. PAGE: DEMAND FORECASTING
# ==========================================
if page == "Demand Forecasting":
    st.title(f"Demand Modeling & Forecast Horizon: {selected_zone}")
    df_hist, df_fcst, metrics, today = load_real_or_mock_demand(selected_zone)

    st.markdown("#### Medallion Pipeline Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='pipeline-card' style='border-color:#CD7F32;'>🥉 <b>Bronze</b><br>Raw Ingestion</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='pipeline-card' style='border-color:#C0C0C0;'>🥈 <b>Silver</b><br>Stationarity & VIF Clear</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='pipeline-card' style='border-color:#FFD700;'>🥇 <b>Gold</b><br>Skew Transformed</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='pipeline-card' style='border-color:#00E676;'>🧠 <b>ML Engine</b><br>AutoML Champion</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FIX 2: Render metrics with explicit light-colored HTML so they are
    #    always readable regardless of Streamlit theme ──────────────────────
    m1, m2, m3 = st.columns(3)

    def render_metric_html(label, value, color="#E2E8F0"):
        return f"""
        <div style="background:#1E2127;border-radius:8px;padding:16px 20px;margin-bottom:4px;">
            <div style="color:#A0AEC0;font-size:0.80rem;font-weight:600;letter-spacing:0.04em;
                        text-transform:uppercase;margin-bottom:6px;">{label}</div>
            <div style="color:{color};font-size:1.55rem;font-weight:700;line-height:1.2;">{value}</div>
        </div>"""

    with m1:
        st.markdown(render_metric_html("Champion Model", metrics["Champion Model"], "#63B3ED"), unsafe_allow_html=True)
    with m2:
        mae_val = f"{metrics['MAE']} ML" if isinstance(metrics['MAE'], float) else metrics['MAE']
        st.markdown(render_metric_html("Mean Absolute Error (MAE)", mae_val, "#68D391"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_metric_html("Seasonality Strength", metrics["Seasonality"], "#F6E05E"), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['Demand'], mode='lines+markers',
                             name='Actuals', line=dict(color='#00E676', width=3)))
    if 'Upper' in df_fcst.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_fcst['Date'], df_fcst['Date'][::-1]]),
            y=pd.concat([df_fcst['Upper'], df_fcst['Lower'][::-1]]),
            fill='toself', fillcolor='rgba(213,0,249,0.2)',
            line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
    fig.add_trace(go.Scatter(x=df_fcst['Date'], y=df_fcst['Demand'], mode='lines+markers',
                             name='7-Day Forecast', line=dict(color='#D500F9', width=3, dash='dash')))
    fig.add_vline(x=today.timestamp() * 1000, line_dash="dot", line_color="white")
    fig.update_layout(plot_bgcolor='#1E2127', paper_bgcolor='#0E1117',
                      font=dict(color='white'), hovermode="x unified",
                      margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. PAGE: SUPPLY OPTIMIZATION (MAP)
# ==========================================
elif page == "Supply Optimization (Map)":
    st.title("Regional Infrastructure & Supply Optimization")

    selected_day = st.slider("Select Optimization Day:", min_value=1, max_value=7, value=1)

    # --- DETERMINE ACTIVE COMPONENTS ---
    active_nodes = set([selected_zone])
    filtered_plan = pd.DataFrame()

    if opt_plan_df is not None:
        day_filter = f"Day {selected_day}"
        filtered_plan = opt_plan_df[
            (opt_plan_df['Day'] == day_filter) &
            (opt_plan_df['Target_Zone'].astype(str).str.contains(selected_zone))
        ]
        active_nodes.update(filtered_plan['Source_UID'].dropna().unique().tolist())

    # --- DYNAMIC STYLING FUNCTIONS ---
    def get_pipe_style(feature):
        props = feature.get('properties', {})
        pipe_type = str(props.get('Pipe_Type', '')).lower()
        start = props.get('Start_Node')
        end   = props.get('End_Node')
        is_active = (start in active_nodes) or (end in active_nodes)
        if not is_active:
            return {'color': '#AAAAAA', 'weight': 1.5, 'opacity': 0.35}
        if 'prod' in pipe_type and 'dsr' in pipe_type:
            color = '#0077CC'
        elif 'dsr' in pipe_type and 'zone' in pipe_type:
            color = '#009944'
        elif 'zone' in pipe_type:
            color = '#9B30FF'
        else:
            color = '#666666'
        return {'color': color, 'weight': 4, 'opacity': 1.0}

    def get_prod_style(feature):
        sid = feature.get('properties', {}).get('Source_ID')
        if sid in active_nodes:
            return {'color': '#005FA3', 'fillColor': '#00BFFF', 'radius': 8, 'fillOpacity': 0.9, 'weight': 2}
        return {'color': '#999999', 'fillColor': '#CCCCCC', 'radius': 4, 'fillOpacity': 0.5, 'weight': 1}

    def get_dsr_style(feature):
        asset = feature.get('properties', {}).get('Asset')
        if asset in active_nodes:
            return {'color': '#CC5500', 'fillColor': '#FF9100', 'radius': 7, 'fillOpacity': 0.9, 'weight': 2}
        return {'color': '#999999', 'fillColor': '#CCCCCC', 'radius': 4, 'fillOpacity': 0.5, 'weight': 1}

    def get_zone_style(feature):
        name = feature.get('properties', {}).get('Name')
        if name == selected_zone:
            return {'color': '#CC0000', 'fillColor': '#FF1744', 'radius': 10, 'fillOpacity': 0.9, 'weight': 2}
        if name in active_nodes:
            return {'color': '#CC2222', 'fillColor': '#FF5555', 'radius': 7, 'fillOpacity': 0.7, 'weight': 1}
        return {'color': '#999999', 'fillColor': '#CCCCCC', 'radius': 4, 'fillOpacity': 0.5, 'weight': 1}

    # ── FIX 3: Light tile map (CartoDB Positron) ─────────────────────────
    m = folium.Map(location=[52.5, -1.5], zoom_start=7, tiles="CartoDB positron")

    # ── FIX 4: Layer-control CSS kept inside the map viewport ─────────────
    m.get_root().html.add_child(folium.Element("""
        <style>
        /* Popup & tooltip styling for light map */
        .leaflet-popup-content-wrapper, .leaflet-tooltip {
            background-color: #FFFFFF !important;
            color: #1A202C !important;
            border: 1px solid #CBD5E0 !important;
            border-radius: 6px !important;
            font-size: 13px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        }
        .leaflet-popup-tip {
            background-color: #FFFFFF !important;
        }
        .leaflet-popup-content table td,
        .leaflet-popup-content table th {
            color: #1A202C !important;
            background-color: #FFFFFF !important;
        }
        /* ── FIX 4: Keep layer-control inside the map container ── */
        .leaflet-top.leaflet-right {
            right: 6px !important;
            top: 6px !important;
            max-height: calc(100% - 12px);
            overflow-y: auto;
        }
        .leaflet-control-layers {
            max-height: 90%;
            overflow-y: auto;
            font-size: 13px !important;
        }
        </style>
    """))

    # Add Layers
    if pipes_gdf is not None:
        try:
            folium.GeoJson(pipes_gdf.to_json(), name="Pipelines",
                           style_function=get_pipe_style,
                           tooltip=folium.GeoJsonTooltip(
                               fields=['Start_Node', 'End_Node', 'Length_Meters'],
                               aliases=['Start:', 'End:', 'Length (m):'])).add_to(m)
        except Exception:
            fields = [c for c in pipes_gdf.columns if c != 'geometry']
            folium.GeoJson(pipes_gdf.to_json(), name="Pipelines",
                           style_function=get_pipe_style,
                           popup=folium.GeoJsonPopup(fields=fields)).add_to(m)

    if prod_gdf is not None:
        try:
            folium.GeoJson(prod_gdf.to_json(), name="Production Plants",
                           marker=folium.CircleMarker(),
                           style_function=get_prod_style,
                           tooltip=folium.GeoJsonTooltip(fields=['Source_ID'], aliases=['Plant:'])).add_to(m)
        except Exception:
            fields = [c for c in prod_gdf.columns if c != 'geometry']
            folium.GeoJson(prod_gdf.to_json(), name="Production Plants",
                           marker=folium.CircleMarker(),
                           style_function=get_prod_style,
                           popup=folium.GeoJsonPopup(fields=fields)).add_to(m)

    if dsr_gdf is not None:
        try:
            folium.GeoJson(dsr_gdf.to_json(), name="DSRs",
                           marker=folium.CircleMarker(),
                           style_function=get_dsr_style,
                           tooltip=folium.GeoJsonTooltip(fields=['Asset'], aliases=['DSR:'])).add_to(m)
        except Exception:
            fields = [c for c in dsr_gdf.columns if c != 'geometry']
            folium.GeoJson(dsr_gdf.to_json(), name="DSRs",
                           marker=folium.CircleMarker(),
                           style_function=get_dsr_style,
                           popup=folium.GeoJsonPopup(fields=fields)).add_to(m)

    if demand_gdf is not None:
        try:
            folium.GeoJson(demand_gdf.to_json(), name="Demand Zones",
                           marker=folium.CircleMarker(),
                           style_function=get_zone_style,
                           tooltip=folium.GeoJsonTooltip(fields=['Name'], aliases=['Zone:'])).add_to(m)
        except Exception:
            fields = [c for c in demand_gdf.columns if c != 'geometry']
            folium.GeoJson(demand_gdf.to_json(), name="Demand Zones",
                           marker=folium.CircleMarker(),
                           style_function=get_zone_style,
                           popup=folium.GeoJsonPopup(fields=fields)).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ── FIX 4: use_container_width equivalent — pass width=None ──────────
    st_folium(m, width=None, height=480, returned_objects=[])

    st.markdown("---")
    st.subheader(f"📋 AI Prescriptive Action Plan: {selected_zone} | Day {selected_day}")

    # --- CALCULATE METRICS AND SHOW CLEAN TABLE ---
    if not filtered_plan.empty:

        costs = pd.to_numeric(filtered_plan['Total_Cost_£'], errors='coerce').fillna(0)
        total_cost = costs.sum()

        is_deficit = filtered_plan['Action'].str.contains('Deficit|Unmet|Emergency|Prescriptive', case=False, na=False)
        is_surplus = filtered_plan['Action'].str.contains('Slack|Surplus|Excess', case=False, na=False)

        deficit_vol = pd.to_numeric(filtered_plan.loc[is_deficit, 'Volume_ML'], errors='coerce').fillna(0).sum()
        surplus_vol = pd.to_numeric(filtered_plan.loc[is_surplus, 'Volume_ML'], errors='coerce').fillna(0).sum()
        net_status  = surplus_vol - deficit_vol

        # ── FIX 1: colored metric HTML for cost & balance ──────────────
        met1, met2 = st.columns(2)

        with met1:
            st.markdown(f"""
            <div style="background:#1E2127;border-radius:8px;padding:16px 20px;">
                <div style="color:#A0AEC0;font-size:0.80rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">
                    Total Daily Cost
                </div>
                <div style="color:#68D391;font-size:1.55rem;font-weight:700;">
                    £{total_cost:,.2f}
                </div>
            </div>""", unsafe_allow_html=True)

        with met2:
            if net_status > 0:
                balance_color, badge_color, badge_text, vol_str = "#68D391", "#276749", "▲ Surplus", f"{net_status:,.2f} ML"
            elif net_status < 0:
                balance_color, badge_color, badge_text, vol_str = "#FC8181", "#742A2A", "▼ Deficit", f"{abs(net_status):,.2f} ML"
            else:
                balance_color, badge_color, badge_text, vol_str = "#E2E8F0", "#2D3748", "● Balanced", "0 ML Variance"

            st.markdown(f"""
            <div style="background:#1E2127;border-radius:8px;padding:16px 20px;">
                <div style="color:#A0AEC0;font-size:0.80rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">
                    Network Balance
                </div>
                <div style="color:{balance_color};font-size:1.55rem;font-weight:700;display:flex;
                            align-items:center;gap:10px;">
                    {vol_str}
                    <span style="background:{badge_color};color:{balance_color};font-size:0.75rem;
                                 padding:3px 10px;border-radius:12px;font-weight:600;">
                        {badge_text}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        display_df = filtered_plan.drop(columns=['Day', 'Target_Zone'], errors='ignore')
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    elif opt_plan_df is not None:
        st.info(f"No active interventions prescribed for {selected_zone} on Day {selected_day}. Demand is met via passive storage.")
    else:
        st.warning("⚠️ `optimized_schedule.csv` not found in `map_data/`.")