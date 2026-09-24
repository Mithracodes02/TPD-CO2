import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import simpson

st.set_page_config(
    page_title="CO2-TPD Advanced Analyzer",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 CO₂-TPD Advanced Analyzer, Benchmarker & Plotter")
st.markdown(
    "Parse multi-block TPD files, customize publication graphics, quantify basicity distribution, "
    "and obtain literature-benchmarked insights for $\\text{CO}_2$ hydrogenation to methanol."
)

def clean_numeric_series(series):
    """Converts a pandas series to numeric floats, handling text, units, and European commas."""
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


# --- SIDEBAR: 1. DATA INPUT & MAPPING ---
st.sidebar.header("1. Data Input & Mapping")
uploaded_file = st.sidebar.file_uploader("Upload TPD Excel (.xlsx / .xls)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()
        engine = "xlrd" if filename.endswith(".xls") else "openpyxl"

        excel_file = pd.ExcelFile(uploaded_file, engine=engine)
        sheet_name = st.sidebar.selectbox("Select Excel Sheet", excel_file.sheet_names)
        
        raw_full = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, engine=engine)
        
        default_header_idx = 23
        for r_idx, row_vals in raw_full.iterrows():
            row_str = " ".join([str(v) for v in row_vals.values if pd.notna(v)])
            if "Temperature" in row_str and "TCD" in row_str:
                default_header_idx = r_idx + 3
                break

        header_row = st.sidebar.number_input("Header Row (0-indexed)", min_value=0, max_value=100, value=default_header_idx)
        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, engine=engine)
        
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

        def_temp_col = 12 if len(col_options) > 12 else 0
        def_tcd_col = 13 if len(col_options) > 13 else min(1, len(col_options)-1)

        temp_col_sel = st.sidebar.selectbox("Temperature Column", col_options, index=def_temp_col)
        tcd_col_sel = st.sidebar.selectbox("TCD Signal Column", col_options, index=def_tcd_col)
        
        temp_idx = int(temp_col_sel.split("Idx ")[1].split(")")[0])
        tcd_idx = int(tcd_col_sel.split("Idx ")[1].split(")")[0])

        data_df = df_raw.iloc[header_row + 1:].copy()

        sample_mass = st.sidebar.number_input("Sample Mass (mg)", min_value=0.1, value=50.0, step=0.1)
        baseline_subtraction = st.sidebar.checkbox("Apply Linear Baseline Correction", value=True)
        
        # --- SIDEBAR: 2. BASIC SITE BOUNDARIES ---
        st.sidebar.header("2. Basic Site Temperature Boundaries")
        weak_max = st.sidebar.number_input("Weak / Medium Boundary (°C)", min_value=50.0, max_value=300.0, value=200.0, step=10.0)
        medium_max = st.sidebar.number_input("Medium / Strong Boundary (°C)", min_value=200.0, max_value=600.0, value=400.0, step=10.0)

        # --- DATA PRE-PROCESSING ---
        clean_df = pd.DataFrame({
            "Temperature": clean_numeric_series(data_df.iloc[:, temp_idx]),
            "TCD": clean_numeric_series(data_df.iloc[:, tcd_idx])
        }).dropna().sort_values("Temperature").reset_index(drop=True)
        
        if len(clean_df) < 2:
            st.error("Not enough numeric data points found. Adjust 'Header Row' or check column mapping.")
            st.stop()

        clean_df["TCD_Norm"] = (clean_df["TCD"] / sample_mass) * 1000
        
        if baseline_subtraction:
            start_val = clean_df["TCD_Norm"].iloc[0]
            end_val = clean_df["TCD_Norm"].iloc[-1]
            baseline = np.linspace(start_val, end_val, len(clean_df))
            clean_df["TCD_Processed"] = (clean_df["TCD_Norm"] - baseline).clip(lower=0)
        else:
            clean_df["TCD_Processed"] = clean_df["TCD_Norm"]

        # --- REGIONAL INTEGRATION ---
        total_area = simpson(y=clean_df["TCD_Processed"].values, x=clean_df["Temperature"].values)
        
        df_weak = clean_df[clean_df["Temperature"] < weak_max]
        df_med = clean_df[(clean_df["Temperature"] >= weak_max) & (clean_df["Temperature"] < medium_max)]
        df_strong = clean_df[clean_df["Temperature"] >= medium_max]

        area_weak = simpson(y=df_weak["TCD_Processed"].values, x=df_weak["Temperature"].values) if len(df_weak) > 1 else 0.0
        area_med = simpson(y=df_med["TCD_Processed"].values, x=df_med["Temperature"].values) if len(df_med) > 1 else 0.0
        area_strong = simpson(y=df_strong["TCD_Processed"].values, x=df_strong["Temperature"].values) if len(df_strong) > 1 else 0.0

        pct_weak = (area_weak / total_area * 100) if total_area > 0 else 0
        pct_med = (area_med / total_area * 100) if total_area > 0 else 0
        pct_strong = (area_strong / total_area * 100) if total_area > 0 else 0

        # --- SIDEBAR: 3. GRAPHIC & AXIS CONTROLS ---
        st.sidebar.header("3. Graph Customization & Axes")
        
        with st.sidebar.expander("✏️ Axis Titles & Font Formatting", expanded=False):
            title_text = st.text_input("Plot Title", "CO₂ Temperature-Programmed Desorption Profile")
            xlabel_text = st.text_input("X-Axis Label", "Temperature (°C)")
            ylabel_text = st.text_input("Y-Axis Label", "TCD Signal (a.u. / g_cat)")
            font_family = st.selectbox("Font Family", ["DejaVu Sans", "DejaVu Serif", "Arial", "Times New Roman", "Courier New"])
            title_size = st.slider("Title Font Size", 8, 24, 14)
            label_size = st.slider("Axis Label Font Size", 8, 20, 12)
            tick_size = st.slider("Tick Font Size", 6, 16, 10)
            font_weight = st.selectbox("Font Weight", ["normal", "bold"])

        with st.sidebar.expander("🎨 Color Customization", expanded=False):
            color_mode = st.radio("Shading Style", ["Shade by Basic Site Regions", "Single Color Peak Fill"])
            curve_color = st.color_picker("Main Curve Line Color", "#1F77B4")
            weak_color = st.color_picker("Weak Sites Fill Color", "#3182BD")
            med_color = st.color_picker("Medium Sites Fill Color", "#E6550D")
            strong_color = st.color_picker("Strong Sites Fill Color", "#DE2D26")
            single_fill_color = st.color_picker("Single Fill Color", "#6BAED6")
            fill_alpha = st.slider("Fill Transparency (Alpha)", 0.0, 1.0, 0.35, step=0.05)

        with st.sidebar.expander("📐 Scale Ranges & Rescaling", expanded=False):
            auto_x = st.checkbox("Auto X-Axis Range", value=True)
            min_temp, max_temp = float(clean_df["Temperature"].min()), float(clean_df["Temperature"].max())
            if not auto_x:
                x_min = st.number_input("X Min (°C)", value=min_temp, step=10.0)
                x_max = st.number_input("X Max (°C)", value=max_temp, step=10.0)
            
            auto_y = st.checkbox("Auto Y-Axis Range", value=True)
            min_tcd, max_tcd = float(clean_df["TCD_Processed"].min()), float(clean_df["TCD_Processed"].max())
            if not auto_y:
                y_min = st.number_input("Y Min", value=0.0, step=0.01)
                y_max = st.number_input("Y Max", value=max_tcd * 1.1, step=0.01)

        with st.sidebar.expander("📏 Ticks, Lines & Canvas Options", expanded=False):
            show_major_ticks = st.checkbox("Show Major Ticks", value=True)
            show_minor_ticks = st.checkbox("Show Minor Ticks", value=True)
            show_grid = st.checkbox("Show Grid Lines", value=False)
            show_box_border = st.checkbox("Box Frame (Top/Right Spines)", value=True)
            line_width = st.slider("Line Width (pt)", 0.5, 4.0, 1.5, step=0.25)
            fig_width = st.slider("Figure Width (in)", 3.0, 10.0, 6.5, step=0.5)
            fig_height = st.slider("Figure Height (in)", 2.5, 8.0, 4.5, step=0.5)
            dpi_val = st.number_input("Export DPI", value=600, step=100)

        # --- PLOTTING ENGINE ---
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

        ax.plot(clean_df["Temperature"], clean_df["TCD_Processed"], color=curve_color, linewidth=line_width, label="CO₂ Signal")

        if color_mode == "Shade by Basic Site Regions":
            ax.fill_between(df_weak["Temperature"], df_weak["TCD_Processed"], color=weak_color, alpha=fill_alpha, label=f"Weak (<{weak_max:.0f}°C)")
            ax.fill_between(df_med["Temperature"], df_med["TCD_Processed"], color=med_color, alpha=fill_alpha, label=f"Medium ({weak_max:.0f}-{medium_max:.0f}°C)")
            ax.fill_between(df_strong["Temperature"], df_strong["TCD_Processed"], color=strong_color, alpha=fill_alpha, label=f"Strong (>{medium_max:.0f}°C)")
            ax.legend(fontsize=tick_size - 1, frameon=False)
        else:
            ax.fill_between(clean_df["Temperature"], clean_df["TCD_Processed"], color=single_fill_color, alpha=fill_alpha)

        ax.set_xlabel(xlabel_text, fontsize=label_size, fontweight=font_weight)
        ax.set_ylabel(ylabel_text, fontsize=label_size, fontweight=font_weight)
        ax.set_title(title_text, fontsize=title_size, fontweight=font_weight)

        # Axis scaling
        if not auto_x:
            ax.set_xlim(x_min, x_max)
        if not auto_y:
            ax.set_ylim(y_min, y_max)

        # Ticks and Grid toggles
        if show_major_ticks:
            ax.tick_params(axis="both", which="major", labelsize=tick_size, bottom=True, left=True)
        else:
            ax.tick_params(axis="both", which="major", bottom=False, left=False, labelbottom=False, labelleft=False)

        if show_minor_ticks and show_major_ticks:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis="both", which="minor", bottom=True, left=True)
        else:
            ax.tick_params(axis="both", which="minor", bottom=False, left=False)

        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.5)

        if not show_box_border:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # --- MAIN DISPLAY LAYOUT ---
        col1, col2 = st.columns([1.6, 1])

        with col1:
            st.pyplot(fig)
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=dpi_val, bbox_inches="tight")
            st.download_button(
                label="📥 Download High-Res Plot (PNG)",
                data=img_buffer.getvalue(),
                file_name="CO2_TPD_Publication_Plot.png",
                mime="image/png"
            )

        with col2:
            st.subheader("📊 Quantified Basic Sites")
            summary_table = pd.DataFrame({
                "Site Type": ["Weak Sites", "Medium Sites", "Strong Sites", "Total Surface Basicity"],
                "Temp Range": [f"< {weak_max:.0f} °C", f"{weak_max:.0f} – {medium_max:.0f} °C", f"> {medium_max:.0f} °C", "Full Profile"],
                "Desorption Area": [f"{area_weak:.2f}", f"{area_med:.2f}", f"{area_strong:.2f}", f"{total_area:.2f}"],
                "Fraction (%)": [f"{pct_weak:.1f} %", f"{pct_med:.1f} %", f"{pct_strong:.1f} %", "100.0 %"]
            })
            st.dataframe(summary_table, hide_index=True, use_container_width=True)

            max_idx = clean_df["TCD_Processed"].idxmax()
            peak_temp = clean_df.loc[max_idx, "Temperature"]
            st.metric("Primary Peak Temperature (T_max)", f"{peak_temp:.1f} °C")
            st.metric("Sample Mass Evaluated", f"{sample_mass} mg")

        # --- LITERATURE COMPARISON & CATALYTIC EVALUATION MODULE ---
        st.markdown("---")
        st.header("🧠 Literature Comparison & Methanol Synthesis Correlation")

        cat_col1, cat_col2 = st.columns(2)

        with cat_col1:
            st.subheader("📌 Main Conclusions from Profile")
            
            # Automated rules derived from heterogeneous catalysis literature for CO2 hydrogenation
            dominant_site = "Weak" if pct_weak >= max(pct_med, pct_strong) else ("Medium" if pct_med >= pct_strong else "Strong")
            
            st.write(f"• **Dominant Surface Species:** The catalyst profile is dominated by **{dominant_site} Basic Sites** ({max(pct_weak, pct_med, pct_strong):.1f}% of total basicity).")
            st.write(f"• **Desorption Thermal Maxima ($T_{{max}}$):** Primary peak occurs at **{peak_temp:.1f} °C**.")
            
            if peak_temp < 200:
                st.write("• **Site Nature:** Primarily weak hydroxyl groups ($\text{OH}^-$) or bicarbonate surface species formed on mild surface sites.")
            elif 200 <= peak_temp <= 400:
                st.write("• **Site Nature:** Predominantly medium-strength metal–oxygen pairs ($\text{Cu-O-Zr}$, $\text{Zn-O-Zr}$, or $\text{In-O}$ oxygen vacancies) coordinating bidentate carbonates.")
            else:
                st.write("• **Site Nature:** Strong isolated low-coordinated oxygen anions ($\text{O}^{2-}$), inducing stable monodentate carbonate formation.")

        with cat_col2:
            st.subheader("💡 Correlation to $\\text{CO}_2/\\text{CO}/\\text{H}_2$ Methanol Synthesis")
            
            if pct_med >= 40.0:
                st.success("🟢 **HIGH CATALYTIC POTENTIAL FOR METHANOL SYNTHESIS**")
                st.write(
                    "**Literature Benchmark:** In $\\text{CO}_2/\\text{CO}/\\text{H}_2$ feed mixtures (e.g., $\\text{Cu/ZnO/ZrO}_2$ or $\\text{In}_2\\text{O}_3$-based systems), "
                    "**medium basic sites ($200-400\\,^\\circ\\text{C}$)** are recognized as the active sites for optimal $\\text{CO}_2$ activation. "
                    "They bind $\\text{CO}_2$ with moderate adsorption enthalpy, stabilizing reaction intermediates ($*\\text{HCOO}$ formate species) "
                    "without poisoning active sites, enabling rapid hydrogenation to $\\text{CH}_3\\text{OH}$."
                )
            elif pct_weak > 50.0:
                st.warning("🟡 **MODERATE ACTIVITY (WEAK BINDING DOMINATED)**")
                st.write(
                    "**Literature Benchmark:** High concentration of weak basic sites ($<200\\,^\\circ\\text{C}$) leads to low $\\text{CO}_2$ residence time "
                    "and quick desorption before hydrogenation by $\\text{H}_2$ occurs. Expect lower single-pass $\\text{CO}_2$ conversion under industrial reaction conditions ($220-260\\,^\\circ\\text{C}$)."
                )
            else:
                st.error("🔴 **POTENTIAL CARBONATE POISONING / HIGH RWGS SELECTIVITY**")
                st.write(
                    "**Literature Benchmark:** Excessive strong basic sites ($>400\\,^\\circ\\text{C}$) hold $\\text{CO}_2$ too strongly, forming rigid monodentate carbonates. "
                    "This often poisons active sites for methanol production, promoting either deep hydrogenation to $\\text{CH}_4$ or increasing Reverse Water-Gas Shift (RWGS) side-reaction to $\\text{CO}$."
                )

    except Exception as e:
        st.error(f"Error executing analysis: {e}")
else:
    st.info("👋 Upload a CO₂-TPD dataset via the sidebar to access custom plotting, quantification, and literature comparison.")
