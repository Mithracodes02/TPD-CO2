import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import simpson

st.set_page_config(
    page_title="CO2-TPD Analyzer & Plotter",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 CO₂-TPD Data Processor, Site Classifier & Plotter")
st.markdown("Instantly parse multi-block Excel files, calculate basic site distributions (Weak, Medium, Strong), and generate publication-ready TPD graphs.")

def clean_numeric_series(series):
    """Converts a pandas series to numeric floats, stripping text, units, and converting European decimal commas."""
    def extract_num(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().replace(',', '.')
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val_str)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return np.nan
        return np.nan

    return series.apply(extract_num)


# --- SIDEBAR: DATA UPLOAD & COLUMN PARSING ---
st.sidebar.header("1. Data Input & Mass Normalization")
uploaded_file = st.sidebar.file_uploader("Upload TPD Excel (.xlsx / .xls)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()
        engine = "xlrd" if filename.endswith(".xls") else "openpyxl"

        excel_file = pd.ExcelFile(uploaded_file, engine=engine)
        sheet_name = st.sidebar.selectbox("Select Excel Sheet", excel_file.sheet_names)
        
        # Read raw preview to locate header
        raw_full = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, engine=engine)
        
        default_header_idx = 23
        for r_idx, row_vals in raw_full.iterrows():
            row_str = " ".join([str(v) for v in row_vals.values if pd.notna(v)])
            if "Temperature" in row_str and "TCD" in row_str:
                default_header_idx = r_idx + 3
                break

        header_row = st.sidebar.number_input("Header Row (0-indexed)", min_value=0, max_value=100, value=default_header_idx)
        
        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, engine=engine)
        
        # Column selection mapping
        header_vals = df_raw.iloc[header_row].values
        col_options = []
        for i, val in enumerate(header_vals):
            col_name = ""
            temp_i = i
            while temp_i >= 0:
                col_name = chr(temp_i % 26 + 65) + col_name
                temp_i = temp_i // 26 - 1
            
            header_str = str(val) if pd.notna(val) else "Unnamed"
            col_options.append(f"Col {col_name} (Idx {i}): {header_str}")

        st.sidebar.subheader("Column Mapping")
        def_temp_col = 12 if len(col_options) > 12 else 0
        def_tcd_col = 13 if len(col_options) > 13 else min(1, len(col_options)-1)

        temp_col_sel = st.sidebar.selectbox("Temperature Column", col_options, index=def_temp_col)
        tcd_col_sel = st.sidebar.selectbox("TCD Signal Column", col_options, index=def_tcd_col)
        
        temp_idx = int(temp_col_sel.split("Idx ")[1].split(")")[0])
        tcd_idx = int(tcd_col_sel.split("Idx ")[1].split(")")[0])

        data_df = df_raw.iloc[header_row + 1:].copy()

        # Sample & Baseline Parameters
        sample_mass = st.sidebar.number_input("Sample Mass (mg)", min_value=0.1, value=50.0, step=0.1)
        baseline_subtraction = st.sidebar.checkbox("Apply Linear Baseline Correction", value=True)
        
        # --- BASIC SITES RANGE CONFIGURATION ---
        st.sidebar.header("2. Basic Site Temperature Boundaries")
        weak_max = st.sidebar.number_input("Weak / Medium Boundary (°C)", min_value=50.0, max_value=300.0, value=200.0, step=10.0)
        medium_max = st.sidebar.number_input("Medium / Strong Boundary (°C)", min_value=200.0, max_value=600.0, value=400.0, step=10.0)
        show_site_regions = st.sidebar.checkbox("Shade Basic Site Regions on Graph", value=True)

        # --- CLEANING & PRE-PROCESSING DATA ---
        clean_df = pd.DataFrame({
            "Temperature": clean_numeric_series(data_df.iloc[:, temp_idx]),
            "TCD": clean_numeric_series(data_df.iloc[:, tcd_idx])
        }).dropna().sort_values("Temperature").reset_index(drop=True)
        
        if len(clean_df) < 2:
            st.error("Not enough numeric data points found. Adjust 'Header Row' to row 23 or check column selections.")
            st.stop()

        # Mass normalization (Signal per gram basis)
        clean_df["TCD_Norm"] = (clean_df["TCD"] / sample_mass) * 1000
        
        # Baseline subtraction
        if baseline_subtraction:
            start_val = clean_df["TCD_Norm"].iloc[0]
            end_val = clean_df["TCD_Norm"].iloc[-1]
            baseline = np.linspace(start_val, end_val, len(clean_df))
            clean_df["TCD_Processed"] = (clean_df["TCD_Norm"] - baseline).clip(lower=0)
        else:
            clean_df["TCD_Processed"] = clean_df["TCD_Norm"]

        # --- REGIONAL INTEGRATION & BASIC SITE ANALYSIS ---
        total_desorption_area = simpson(y=clean_df["TCD_Processed"].values, x=clean_df["Temperature"].values)
        
        df_weak = clean_df[clean_df["Temperature"] < weak_max]
        df_med = clean_df[(clean_df["Temperature"] >= weak_max) & (clean_df["Temperature"] < medium_max)]
        df_strong = clean_df[clean_df["Temperature"] >= medium_max]

        area_weak = simpson(y=df_weak["TCD_Processed"].values, x=df_weak["Temperature"].values) if len(df_weak) > 1 else 0.0
        area_med = simpson(y=df_med["TCD_Processed"].values, x=df_med["Temperature"].values) if len(df_med) > 1 else 0.0
        area_strong = simpson(y=df_strong["TCD_Processed"].values, x=df_strong["Temperature"].values) if len(df_strong) > 1 else 0.0

        pct_weak = (area_weak / total_desorption_area * 100) if total_desorption_area > 0 else 0
        pct_med = (area_med / total_desorption_area * 100) if total_desorption_area > 0 else 0
        pct_strong = (area_strong / total_desorption_area * 100) if total_desorption_area > 0 else 0

        # --- GRAPH CUSTOMIZATION CONTROLS ---
        st.sidebar.header("3. Publication Graph Settings")
        
        LAYOUT_PRESETS = [
            "1. Nature / Science (Minimalist Serif)",
            "2. ACS Catalysis (Classic Bold Standard)",
            "3. Elsevier / Calphad (Clean Sans-Serif)",
            "4. Dark High-Contrast (Presentation)",
            "5. Royal Blue & Steel (Modern)",
            "6. Emerald / Forest Accent",
            "7. Crimson Accent",
            "8. Greyscale Monochrome (Print Safe)",
            "9. Vibrant Rainbow Spectrum",
            "10. Boxed Border Classic (Analytical)"
        ]
        
        selected_preset = st.sidebar.selectbox("Select Graph Style Preset", LAYOUT_PRESETS)
        
        with st.sidebar.expander("Axis & Font Customization"):
            font_family = st.selectbox("Font Family", ["DejaVu Sans", "DejaVu Serif", "Arial", "Times New Roman", "Courier New"])
            title_size = st.slider("Title Font Size", 8, 24, 14)
            label_size = st.slider("Axis Label Font Size", 8, 20, 12)
            tick_size = st.slider("Tick Font Size", 6, 16, 10)
            font_weight = st.selectbox("Font Weight", ["normal", "bold"])
            line_width = st.slider("Line Width (pt)", 0.5, 4.0, 1.5, step=0.25)
            fig_width = st.slider("Figure Width (inches)", 3.0, 10.0, 6.0, step=0.5)
            fig_height = st.slider("Figure Height (inches)", 2.5, 8.0, 4.5, step=0.5)
            dpi_val = st.number_input("Export Resolution (DPI)", value=600, step=100)

        # --- PLOTTING ENGINE ---
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
        
        style_color = "#1f77b4"
        bg_color = "white"
        show_box = True
        grid_enabled = False
        
        if "Nature" in selected_preset:
            font_family = "DejaVu Serif"
            style_color = "#2b2b2b"
            show_box = False
        elif "ACS" in selected_preset:
            font_family = "DejaVu Sans"
            style_color = "#b22222"
            font_weight = "bold"
        elif "Elsevier" in selected_preset:
            font_family = "DejaVu Sans"
            style_color = "#005580"
        elif "Dark" in selected_preset:
            bg_color = "#121212"
            style_color = "#00e676"
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.tick_params(colors="white")
        elif "Royal Blue" in selected_preset:
            style_color = "#4169E1"
            grid_enabled = True
        elif "Emerald" in selected_preset:
            style_color = "#00875A"
        elif "Crimson" in selected_preset:
            style_color = "#DC143C"
        elif "Greyscale" in selected_preset:
            style_color = "#000000"
        elif "Vibrant" in selected_preset:
            style_color = "#FF4500"
        elif "Boxed" in selected_preset:
            style_color = "#111111"
            grid_enabled = True

        plt.rcParams["font.family"] = font_family

        # Main TPD curve
        ax.plot(clean_df["Temperature"], clean_df["TCD_Processed"], color=style_color, linewidth=line_width, label="CO₂ Desorption Signal")
        
        # Shade Basic Site Regions
        if show_site_regions:
            ax.fill_between(df_weak["Temperature"], df_weak["TCD_Processed"], color="#1f77b4", alpha=0.3, label="Weak Sites (<200°C)")
            ax.fill_between(df_med["Temperature"], df_med["TCD_Processed"], color="#ff7f0e", alpha=0.3, label="Medium Sites (200-400°C)")
            ax.fill_between(df_strong["Temperature"], df_strong["TCD_Processed"], color="#d62728", alpha=0.3, label="Strong Sites (>400°C)")
            ax.legend(fontsize=tick_size - 1, frameon=False)
        else:
            ax.fill_between(clean_df["Temperature"], clean_df["TCD_Processed"], color=style_color, alpha=0.15)

        ax.set_xlabel("Temperature (°C)", fontsize=label_size, fontweight=font_weight)
        ax.set_ylabel("TCD Signal (a.u. / g_cat)", fontsize=label_size, fontweight=font_weight)
        ax.set_title("CO₂ Temperature-Programmed Desorption Profile", fontsize=title_size, fontweight=font_weight)
        
        ax.tick_params(axis="both", which="major", labelsize=tick_size)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        if grid_enabled:
            ax.grid(True, linestyle="--", alpha=0.5)

        if not show_box:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # --- DASHBOARD DISPLAY ---
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.pyplot(fig)
            
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=dpi_val, bbox_inches="tight")
            st.download_button(
                label="📥 Download High-Res Plot (PNG)",
                data=img_buffer.getvalue(),
                file_name="CO2_TPD_Basic_Sites_Graph.png",
                mime="image/png"
            )

        with col2:
            st.subheader("📊 Basic Sites Distribution")
            
            summary_table = pd.DataFrame({
                "Basic Site Type": ["Weak Sites", "Medium Sites", "Strong Sites", "Total"],
                "Temperature Range": [f"< {weak_max:.0f} °C", f"{weak_max:.0f} – {medium_max:.0f} °C", f"> {medium_max:.0f} °C", "Full Spectrum"],
                "Desorption Area (a.u.*°C/g)": [f"{area_weak:.2f}", f"{area_med:.2f}", f"{area_strong:.2f}", f"{total_desorption_area:.2f}"],
                "Distribution (%)": [f"{pct_weak:.1f} %", f"{pct_med:.1f} %", f"{pct_strong:.1f} %", "100.0 %"]
            })
            
            st.dataframe(summary_table, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            max_idx = clean_df["TCD_Processed"].idxmax()
            peak_temp = clean_df.loc[max_idx, "Temperature"]
            st.metric("Desorption Peak Temperature (T_max)", f"{peak_temp:.1f} °C")
            st.metric("Sample Weight Used", f"{sample_mass} mg")

    except Exception as e:
        st.error(f"Error processing data file: {e}")
else:
    st.info("👋 Upload a CO₂-TPD `.xlsx` or `.xls` data file via the sidebar to get started.")
