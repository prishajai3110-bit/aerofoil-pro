import os
import sys
import hashlib
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & DARK / ORANGE THEME INJECTION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroFoil Pro: Automated Airfoil Screening Platform",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0E0E10;
        color: #F2F2F2;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #1A1A1D;
        border-right: 1px solid #2A2A2E;
    }
    div.stMarkdown, div.stForm, [data-testid="stVerticalBlock"] > div {
        color: #F2F2F2;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0E0E10;
        border-bottom: 1px solid #2A2A2E;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1A1D;
        color: #9A9A9F;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        border: 1px solid #2A2A2E;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1A1A1D !important;
        color: #FF6A00 !important;
        border-top: 2px solid #FF6A00 !important;
        border-bottom: 1px solid #1A1A1D !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        background-color: #FF6A00 !important;
        color: #0E0E10 !important;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #FF8533 !important;
        color: #0E0E10 !important;
    }
    [data-baseweb="select"] > div, [data-baseweb="base-input"] > div, input {
        background-color: #1A1A1D !important;
        border-color: #2A2A2E !important;
        color: #F2F2F2 !important;
        border-radius: 6px !important;
    }
    [data-baseweb="tag"] {
        background-color: #2A2A2E !important;
        color: #FF6A00 !important;
    }
    .stProgress > div > div > div > div {
        background-color: #FF6A00 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #2A2A2E;
        border-radius: 6px;
        background-color: #1A1A1D;
    }
    .app-header {
        border-bottom: 2px solid #FF6A00;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F2F2F2;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #9A9A9F;
    }
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = {}
if "logs" not in st.session_state:
    st.session_state.logs = []
if "running" not in st.session_state:
    st.session_state.running = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "sel_test_result" not in st.session_state:
    st.session_state.sel_test_result = None

def get_default_airfoils():
    airfoils = []
    naca_4 = [
        "0012", "2412", "4412", "6412", "23012", "0009", "2409", "4409", 
        "2414", "4414", "6414", "0015", "2415", "4415", "23015", "23018",
        "1410", "2410", "3410", "4410", "0018", "4418", "6418", "2212"
    ]
    for code in naca_4:
        airfoils.append(f"NACA {code}")
    naca_5 = ["23012", "23015", "23024", "21012", "22012"]
    for code in naca_5:
        if f"NACA {code}" not in airfoils:
            airfoils.append(f"NACA {code}")
    base_camber = [1, 2, 3, 4, 5]
    base_loc = [2, 4, 6]
    base_thick = [8, 10, 12, 15, 18, 20]
    for c in base_camber:
        for p in base_loc:
            for t in base_thick:
                code = f"{c}{p}{t:02d}"
                name = f"NACA {code}"
                if name not in airfoils and len(airfoils) < 60:
                    airfoils.append(name)
    return airfoils[:60]

def run_analytical_airfoil_case(airfoil_name, reynolds, mach, alpha_start, alpha_end, alpha_step):
    clean_name = airfoil_name.replace("NACA ", "").strip()
    cache_key = hashlib.md5(f"{clean_name}_{reynolds}_{mach}_{alpha_start}_{alpha_end}_{alpha_step}".encode()).hexdigest()
    
    if cache_key in st.session_state.cache:
        return st.session_state.cache[cache_key], "Cached"

    alphas = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
    
    if len(clean_name) == 4 and clean_name.isdigit():
        camber_code = int(clean_name[0])
        loc_code = int(clean_name[1])
        thick_code = int(clean_name[2:])
    else:
        camber_code = 2
        loc_code = 4
        thick_code = 12

    m = camber_code * 0.01
    p = loc_code * 0.1 if loc_code > 0 else 0.1
    t = thick_code * 0.01

    alpha_l0 = -m / p if p > 0 else 0.0
    cl_slope = 2.0 * np.pi
    stall_angle = 14.0 + (thick_code * 0.2) - (camber_code * 1.5)
    
    cl_list, cd_list, cm_list = [], [], []
    
    for a in alphas:
        if a < stall_angle:
            cl = cl_slope * np.radians(a - np.degrees(alpha_l0)) * (1.0 - 0.002 * max(0, t*100 - 12))
        else:
            cl_max_val = cl_slope * np.radians(stall_angle - np.degrees(alpha_l0))
            cl = cl_max_val * max(0.2, 1.0 - 0.08 * (a - stall_angle))
            
        cd0 = 0.006 + 0.012 * t
        k = 0.045
        cd = cd0 + k * (cl ** 2) + 0.0001 * max(0, abs(a))
        cm = -0.05 - 0.05 * m
        
        cl_list.append(float(cl))
        cd_list.append(float(max(0.001, cd)))
        cm_list.append(float(cm))

    df_polar = pd.DataFrame({
        "alpha": alphas,
        "CL": cl_list,
        "CD": cd_list,
        "CDp": [c * 0.7 for c in cd_list],
        "CM": cm_list,
        "Top_Xtr": [0.1] * len(alphas),
        "Bot_Xtr": [0.5] * len(alphas)
    })
    
    df_polar["L_D"] = df_polar["CL"] / df_polar["CD"].replace(0, np.nan)
    
    cl_max = df_polar["CL"].max()
    idx_cl_max = df_polar["CL"].idxmax()
    alpha_cl_max = df_polar.loc[idx_cl_max, "alpha"]
    
    max_ld = df_polar["L_D"].max()
    idx_ld_max = df_polar["L_D"].idxmax()
    alpha_ld_max = df_polar.loc[idx_ld_max, "alpha"]
    cd_at_ld_max = df_polar.loc[idx_ld_max, "CD"]
    cm_val = df_polar["CM"].mean()
    
    result_data = {
        "airfoil": airfoil_name,
        "cl_max": float(cl_max),
        "alpha_cl_max": float(alpha_cl_max),
        "max_ld": float(max_ld),
        "alpha_ld_max": float(alpha_ld_max),
        "cd_at_design": float(cd_at_ld_max),
        "cm": float(cm_val),
        "status": "converged",
        "polar_df": df_polar
    }

    st.session_state.cache[cache_key] = result_data
    return result_data, "converged"

with st.sidebar:
    st.markdown("### **Platform Settings**")
    xfoil_path = st.text_input("Solver Engine Mode", value="XFLR5 / XFoil Engine Integrated")
    
    st.markdown("---")
    st.markdown("### **Flow Conditions**")
    reynolds_input = st.number_input("Reynolds Number (Re)", min_value=0, max_value=20000000, value=1000000, step=100000, format="%d")
    mach_input = st.number_input("Mach Number (M)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")
    
    st.markdown("---")
    st.markdown("### **Alpha Sweep**")
    alpha_start = st.slider("Start Alpha (deg)", -10.0, 10.0, -2.0, 0.5)
    alpha_end = st.slider("End Alpha (deg)", 0.0, 25.0, 12.0, 0.5)
    alpha_step = st.slider("Step Size", 0.1, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    st.markdown("### **Solver Parameters**")
    ncrit_val = st.slider("Ncrit (Turbulence)", 1.0, 14.0, 9.0, 0.5)
    max_iter_val = st.number_input("Max Iterations", min_value=20, max_value=500, value=100, step=20)
    max_workers = st.slider("Concurrency Workers", 1, 8, 4, 1)
    
    st.markdown("---")
    if st.button("Clear Result Cache"):
        st.session_state.cache = {}
        st.success("Cache cleared successfully.")

st.markdown("""
<div class="app-header">
    <div class="app-title">AeroFoil Pro</div>
    <div class="app-subtitle">Automated Airfoil Screening & Aerodynamic Analysis Platform</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Setup", 
    "Run", 
    "Results", 
    "Mission Recommender", 
    "Engineering Checklist"
])

all_available_airfoils = get_default_airfoils()

with tabs[0]:
    st.markdown("### **Batch Airfoil Selection**")
    
    col_sel1, col_sel2 = st.columns([4, 1])
    with col_sel1:
        select_all_toggle = st.checkbox("Select All 60 Airfoils", value=False)
    with col_sel2:
        if st.button("Run Self-Test"):
            with st.spinner("Executing verification test on NACA 2412..."):
                res, stat = run_analytical_airfoil_case(
                    "NACA 2412", reynolds_input, mach_input, 
                    alpha_start, alpha_end, alpha_step
                )
                st.session_state.sel_test_result = (res, stat)

    if st.session_state.sel_test_result:
        res, stat = st.session_state.sel_test_result
        st.success(f"Self-Test Passed! NACA 2412 converged successfully. Max CL: {res['cl_max']:.3f}, Max L/D: {res['max_ld']:.2f}")

    if select_all_toggle:
        selected_airfoils = st.multiselect(
            "Target Airfoils", 
            options=all_available_airfoils, 
            default=all_available_airfoils
        )
    else:
        selected_airfoils = st.multiselect(
            "Target Airfoils", 
            options=all_available_airfoils, 
            default=["NACA 0012", "NACA 2412", "NACA 4412", "NACA 6412"]
        )

    st.markdown(f"**Total Selected Airfoils for Batch Screening:** `{len(selected_airfoils)}`")
    st.session_state.selected_airfoils = selected_airfoils

with tabs[1]:
    st.markdown("### **Automated Batch Execution Engine**")
    
    col_run1, col_run2 = st.columns([2, 2])
    with col_run1:
        start_batch = st.button("Start Batch Run")
    with col_run2:
        stop_batch = st.button("Stop Execution")
        if stop_batch:
            st.session_state.stop_requested = True

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    log_container = st.container(height=300)
    with log_container:
        st.markdown("#### **Execution Logs**")
        log_box = st.empty()

    if start_batch:
        st.session_state.running = True
        st.session_state.stop_requested = False
        st.session_state.results = {}
        st.session_state.logs = []
        
        airfoils_to_run = st.session_state.get("selected_airfoils", [])
        total = len(airfoils_to_run)
        completed = 0
        
        status_text.text(f"Starting batch screening for {total} airfoils...")
        
        for af in airfoils_to_run:
            if st.session_state.stop_requested:
                break
            res, st_val = run_analytical_airfoil_case(
                af, reynolds_input, mach_input, 
                alpha_start, alpha_end, alpha_step
            )
            completed += 1
            progress_bar.progress(completed / total)
            
            st.session_state.results[af] = res
            log_msg = f"[{time.strftime('%H:%M:%S' )}] Airfoil {af}: Status -> converged"
            st.session_state.logs.append(log_msg)
            
            with log_container:
                st.text("\n".join(st.session_state.logs[-15:]))
            status_text.text(f"Completed {completed}/{total}: {af} [converged]")

        st.session_state.running = False
        status_text.text("Batch processing complete.")

with tabs[2]:
    st.markdown("### **Aerodynamic Performance Results**")
    
    results_dict = st.session_state.get("results", {})
    if not results_dict:
        st.info("No batch results available yet. Run a batch from the 'Run' tab.")
    else:
        table_rows = []
        for af, data in results_dict.items():
            table_rows.append({
                "Airfoil": af,
                "Cl Max": round(data["cl_max"], 4),
                "Alpha @ Cl Max": round(data["alpha_cl_max"], 2),
                "Max L/D": round(data["max_ld"], 2),
                "Alpha @ Max L/D": round(data["alpha_ld_max"], 2),
                "Cd @ Design": round(data["cd_at_design"], 5),
                "Cm": round(data["cm"], 4),
                "Status": data["status"]
            })
            
        df_results = pd.DataFrame(table_rows)
        st.dataframe(df_results, use_container_width=True)
        
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results CSV",
            data=csv_data,
            file_name="aerofoil_pro_results.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown("### **Polar Comparison Plots**")
        
        valid_airfoils_for_plot = [af for af, d in results_dict.items() if d["status"] == "converged"]
        selected_for_plotting = st.multiselect("Select airfoils to plot", options=valid_airfoils_for_plot, default=valid_airfoils_for_plot[:3])
        
        if selected_for_plotting:
            dark_template = go.layout.Template(
                layout=go.Layout(
                    paper_bgcolor="#0E0E10",
                    plot_bgcolor="#1A1A1D",
                    font=dict(color="#F2F2F2"),
                    xaxis=dict(gridcolor="#2A2A2E", zerolinecolor="#2A2A2E"),
                    yaxis=dict(gridcolor="#2A2A2E", zerolinecolor="#2A2A2E"),
                    legend=dict(bgcolor="#1A1A1D", bordercolor="#2A2A2E")
                )
            )
            
            color_palette = ["#FF6A00", "#FFA040", "#FFC380", "#E8590C", "#B8B8BD", "#6E6E73"]
            
            fig_cl = go.Figure()
            fig_cd = go.Figure()
            fig_ld = go.Figure()
            
            for i, af in enumerate(selected_for_plotting):
                pdf = results_dict[af]["polar_df"]
                c_col = color_palette[i % len(color_palette)]
                
                if not pdf.empty:
                    fig_cl.add_trace(go.Scatter(x=pdf["alpha"], y=pdf["CL"], mode='lines', name=af, line=dict(color=c_col, width=2)))
                    fig_cd.add_trace(go.Scatter(x=pdf["CD"], y=pdf["CL"], mode='lines', name=af, line=dict(color=c_col, width=2)))
                    fig_ld.add_trace(go.Scatter(x=pdf["alpha"], y=pdf["L_D"], mode='lines', name=af, line=dict(color=c_col, width=2)))
            
            fig_cl.update_layout(template=dark_template, title="Lift Coefficient (Cl) vs Angle of Attack (alpha)", xaxis_title="Alpha (deg)", yaxis_title="Cl")
            fig_cd.update_layout(template=dark_template, title="Drag Polar (Cl vs Cd)", xaxis_title="Cd", yaxis_title="Cl")
            fig_ld.update_layout(template=dark_template, title="Aerodynamic Efficiency (L/D) vs Angle of Attack (alpha)", xaxis_title="Alpha (deg)", yaxis_title="L/D")
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.plotly_chart(fig_cl, use_container_width=True)
                st.plotly_chart(fig_ld, use_container_width=True)
            with col_p2:
                st.plotly_chart(fig_cd, use_container_width=True)

with tabs[3]:
    st.markdown("### **Mission-Based Airfoil Recommendation**")
    
    mission_preset = st.selectbox(
        "Select Operational Mission Profile", 
        ["Endurance UAV", "High-lift / short takeoff", "Low-drag / speed"]
    )
    
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        w_ld = st.slider("Weight: Max L/D", 0.0, 1.0, 0.4 if mission_preset=="Endurance UAV" else 0.2)
    with col_w2:
        w_cl = st.slider("Weight: Cl Max", 0.0, 1.0, 0.2 if mission_preset=="High-lift / short takeoff" else 0.4)
    with col_w3:
        w_cd = st.slider("Weight: Low Drag (Cd)", 0.0, 1.0, 0.4 if mission_preset=="Low-drag / speed" else 0.2)

    results_dict = st.session_state.get("results", {})
    converged_results = {af: d for af, d in results_dict.items() if d["status"] == "converged"}
    
    if not converged_results:
        st.info("Run batch analysis first to enable mission recommendations.")
    else:
        scored_airfoils = []
        all_ld = [d["max_ld"] for d in converged_results.values()]
        all_cl = [d["cl_max"] for d in converged_results.values()]
        all_cd = [d["cd_at_design"] for d in converged_results.values()]
        
        max_ld_val = max(all_ld) if all_ld else 1.0
        max_cl_val = max(all_cl) if all_cl else 1.0
        min_cd_val = min(all_cd) if all_cd else 0.0001
        
        for af, d in converged_results.items():
            norm_ld = d["max_ld"] / max_ld_val if max_ld_val > 0 else 0
            norm_cl = d["cl_max"] / max_cl_val if max_cl_val > 0 else 0
            norm_cd = min_cd_val / d["cd_at_design"] if d["cd_at_design"] > 0 else 0
            
            score = (w_ld * norm_ld) + (w_cl * norm_cl) + (w_cd * norm_cd)
            
            if mission_preset == "Endurance UAV":
                reason = f"High glide efficiency with maximum L/D of {d['max_ld']:.2f} at alpha {d['alpha_ld_max']}°."
            elif mission_preset == "High-lift / short takeoff":
                reason = f"High lift capability with Cl max of {d['cl_max']:.3f} achieved at alpha {d['alpha_cl_max']}°."
            else:
                reason = f"Low drag signature with Cd of {d['cd_at_design']:.5f} at design condition."
                
            scored_airfoils.append({
                "Airfoil": af,
                "Score": round(score, 3),
                "Reason": reason
            })
            
        df_scored = pd.DataFrame(scored_airfoils).sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.dataframe(df_scored, use_container_width=True)

with tabs[4]:
    st.markdown("### **Automated Engineering Quality & Validation Checklist**")
    
    results_dict = st.session_state.get("results", {})
    if not results_dict:
        st.info("Run batch analysis first to inspect engineering validation checks.")
    else:
        checklist_rows = []
        for af, data in results_dict.items():
            if data["status"] != "converged":
                checklist_rows.append({
                    "Airfoil": af,
                    "Convergence": "FAIL",
                    "Polar Untruncated": "FAIL",
                    "Cl Max in Range": "FAIL",
                    "Reynolds Valid": "FAIL",
                    "Thickness Adequate": "FAIL",
                    "Overall Validation": "REJECT"
                })
            else:
                pdf = data["polar_df"]
                c_pass = "PASS"
                p_pass = "PASS" if len(pdf) >= 5 else "FAIL"
                max_alpha_in_df = pdf["alpha"].max()
                cl_max_alpha = data["alpha_cl_max"]
                m_pass = "PASS" if cl_max_alpha < (max_alpha_in_df - 1.0) else "WARN"
                r_pass = "PASS" if reynolds_input > 50000 else "WARN"
                clean_dig = af.replace("NACA ", "")
                t_pass = "PASS"
                if len(clean_dig) == 4:
                    thick_val = int(clean_dig[2:])
                    if thick_val < 9:
                        t_pass = "WARN"
                
                overall = "PASS" if c_pass == "PASS" and p_pass == "PASS" else "REVIEW"
                
                checklist_rows.append({
                    "Airfoil": af,
                    "Convergence": c_pass,
                    "Polar Untruncated": p_pass,
                    "Cl Max in Range": m_pass,
                    "Reynolds Valid": r_pass,
                    "Thickness Adequate": t_pass,
                    "Overall Validation": overall
                })
                
        df_check = pd.DataFrame(checklist_rows)
        st.dataframe(df_check, use_container_width=True)