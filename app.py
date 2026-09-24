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

st.title("🧪 CO₂-TPD Data Processor & Plotter")
st.markdown("Instantly parse Excel files, calculate desorption metrics, and generate publication-ready TCD vs. Temperature figures.")

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
        # Determine engine based on extension
        filename = uploaded_file.name.lower()
        engine = "xlrd" if filename.endswith(".xls") else "openpyxl"

        # Read sheet list
        excel_file = pd.ExcelFile(uploaded_file, engine=engine)
        sheet_name = st.sidebar.selectbox("Select Excel Sheet", excel_file.sheet_names)
        
        # Read raw preview to auto-suggest header row
        raw_preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=40, engine=engine)
        
        # Auto-detect row with multiple numerical values
        suggested_header = 0
        for i, row in raw_preview.iterrows():
            num_count = sum(clean_numeric_series(row).notna())
            if num_count >= 2:
                suggested_header = max(0, i - 1) if i > 0 else 0
                break

        header_row = st.sidebar.number_input("Header Row (0-indexed)", min_value=0, max_value=100, value=suggested_header)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row, engine=engine)
        
        if df.empty:
            st.error("The selected sheet or header row resulted in an empty table. Please adjust the Header Row index.")
            st.stop()

        st.sidebar.subheader("Column Mapping")
        all_cols = [str(c) for c in df.columns]
        
        # Smart detection for temperature and signal columns
        temp_default = next((i for i, c in enumerate(all_cols) if any(k in c.lower() for k in ["temp", "t (°c)", "t/°c", "temperature"])), 0)
        tcd_default = next((i for i, c in enumerate(all_cols) if any(k in c.lower() for k in ["tcd", "signal", "ms", "mv", "intensity"])), min(1, len(all_cols)-1))
        
        temp_col = st.sidebar.selectbox("Temperature Column", all_cols, index=temp_default)
        tcd_col = st.sidebar.selectbox("TCD Signal Column", all_cols, index=tcd_default)
        
        # Raw Data Inspection Box
        with st.sidebar.expander("🔍 Inspect Raw Selected Columns", expanded=False):
            st.write("First 15 raw rows:")
            st.dataframe(df[[temp_col, tcd_col]].head(15))
            
            raw_t = clean_numeric_series(df[temp_col])
            raw_s = clean_numeric_series(df[tcd_col])
            valid_pts = (raw_t.notna() & raw_s.notna()).sum()
            st.info(f"Valid numeric rows found: **{valid_pts}**")

        # Sample & Baseline Parameters
        sample_mass = st.sidebar.number_input("Sample Mass (mg)", min_value=0.1, value=100.0, step=1.0)
        baseline_subtraction = st.sidebar.checkbox("Apply Linear Baseline Correction", value=True)
        
        # --- CLEANING & PRE-PROCESSING DATA ---
        clean_df = pd.DataFrame({
            "Temperature": clean_numeric_series(df[temp_col]),
            "TCD": clean_numeric_series(df[tcd_col])
        })
        
        clean_df = clean_df.dropna().sort_values("Temperature").reset_index(drop=True)
        
        if len(clean_df) < 2:
            st.error("Not enough numeric data points found. Check 'Inspect Raw Selected Columns' in the sidebar to ensure your chosen Header Row and Column selections contain actual numeric numbers.")
            st.stop()

        # Mass normalization (Signal per gram basis)
        clean_df["TCD_Norm"] = (clean_df["TCD"] / sample_mass) * 1000
        
        # Linear baseline subtraction
        if baseline_subtraction:
            start_val = clean_df["TCD_Norm"].iloc[0]
            end_val = clean_df["TCD_Norm"].iloc[-1]
            baseline = np.linspace(start_val, end_val, len(clean_df))
            clean_df["TCD_Processed"] = (clean_df["TCD_Norm"] - baseline).clip(lower=0)
        else:
            clean_df["TCD_Processed"] = clean_df["TCD_Norm"]

        # Simpson Integration
        total_desorption_area = simpson(y=clean_df["TCD_Processed"].values, x=clean_df["Temperature"].values)

        # --- GRAPH CUSTOMIZATION CONTROLS ---
        st.sidebar.header("2. Publication Graph Settings")
        
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

        ax.plot(clean_df["Temperature"], clean_df["TCD_Processed"], color=style_color, linewidth=line_width, label="CO₂ Desorption")
        ax.fill_between(clean_df["Temperature"], clean_df["TCD_Processed"], color=style_color, alpha=0.15)
        
        ax.set_xlabel("Temperature (°C)", fontsize=label_size, fontweight=font_weight)
        ax.set_ylabel("TCD Signal (a.u. / g_cat)", fontsize=label_size, fontweight=font_weight)
        ax.set_title("CO₂ Temperature-Programmed Desorption", fontsize=title_size, fontweight=font_weight)
        
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
                file_name="CO2_TPD_Publication_Graph.png",
                mime="image/png"
            )

        with col2:
            st.subheader("📊 Desorption Metrics")
            st.metric("Total Integrated Area", f"{total_desorption_area:.2f} a.u.*°C/g")
            
            max_idx = clean_df["TCD_Processed"].idxmax()
            peak_temp = clean_df.loc[max_idx, "Temperature"]
            st.metric("Peak Temperature (T_max)", f"{peak_temp:.1f} °C")
            st.metric("Sample Weight Used", f"{sample_mass} mg")
            
            st.markdown("---")
            st.subheader("📋 Parsed Data Sample")
            st.dataframe(clean_df[["Temperature", "TCD_Processed"]].head(10), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing data file: {e}")
else:
    st.info("👋 Upload a CO₂-TPD `.xlsx` or `.xls` data file via the sidebar to get started.")
