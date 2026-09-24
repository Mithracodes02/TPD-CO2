import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import simpson
from scipy.optimize import curve_fit

# Export Libraries
from docx import Document
from docx.shared import Inches
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(
    page_title="CO2-TPD Advanced Analyzer & Peak Fitting",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 CO₂-TPD Analyzer with Gaussian Peak Deconvolution")
st.markdown(
    "Parse TPD datasets, deconvolute overlapping desorption peaks using **Gaussian Fitting**, "
    "customize publication typography, and export reports in **PDF**, **Excel**, or **Word**."
)

def clean_numeric_series(series):
    """Converts pandas series to numeric floats, handling text, units, and European commas."""
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

def gaussian(x, amp, center, sigma):
    """Gaussian function for peak fitting."""
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def multi_gaussian(x, *params):
    """Sum of multiple Gaussian functions."""
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        center = params[i+1]
        sigma = params[i+2]
        y += gaussian(x, amp, center, sigma)
    return y


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
        
        # Temperature Range Truncation (Crucial to exclude high-temp bulk decomposition)
        st.sidebar.subheader("Temperature Data Trimming")
        t_min_fit = st.sidebar.number_input("Min Analysis Temp (°C)", value=50.0, step=10.0)
        t_max_fit = st.sidebar.number_input("Max Analysis Temp (°C)", value=500.0, step=10.0)

        # --- DATA PRE-PROCESSING ---
        clean_df = pd.DataFrame({
            "Temperature": clean_numeric_series(data_df.iloc[:, temp_idx]),
            "TCD": clean_numeric_series(data_df.iloc[:, tcd_idx])
        }).dropna().sort_values("Temperature").reset_index(drop=True)
        
        # Trim data to specified analysis range
        clean_df = clean_df[(clean_df["Temperature"] >= t_min_fit) & (clean_df["Temperature"] <= t_max_fit)].reset_index(drop=True)

        if len(clean_df) < 5:
            st.error("Not enough data points in selected temperature range. Adjust 'Min/Max Analysis Temp'.")
            st.stop()

        clean_df["TCD_Norm"] = (clean_df["TCD"] / sample_mass) * 1000
        
        # Baseline subtraction
        start_val = clean_df["TCD_Norm"].iloc[0]
        end_val = clean_df["TCD_Norm"].iloc[-1]
        baseline = np.linspace(start_val, end_val, len(clean_df))
        clean_df["TCD_Processed"] = (clean_df["TCD_Norm"] - baseline).clip(lower=0)

        # --- SIDEBAR: 2. GAUSSIAN DECONVOLUTION SETTINGS ---
        st.sidebar.header("2. Gaussian Peak Deconvolution")
        enable_fit = st.sidebar.checkbox("Enable Gaussian Deconvolution", value=True)
        num_peaks = st.sidebar.slider("Number of Peaks to Fit", min_value=1, max_value=4, value=3)

        initial_guesses = []
        bounds_lower = []
        bounds_upper = []

        default_centers = [150.0, 300.0, 450.0, 550.0]
        for p in range(num_peaks):
            with st.sidebar.expander(f"Peak {p+1} Initial Parameters", expanded=(p == 0)):
                guess_amp = st.number_input(f"P{p+1} Amplitude Guess", value=float(clean_df["TCD_Processed"].max() * 0.5), key=f"amp_{p}")
                guess_center = st.number_input(f"P{p+1} Center Temp (°C)", value=default_centers[p], key=f"cen_{p}")
                guess_sigma = st.number_input(f"P{p+1} Sigma (°C)", value=30.0, key=f"sig_{p}")
                
                initial_guesses.extend([guess_amp, guess_center, guess_sigma])
                bounds_lower.extend([0.0, t_min_fit, 5.0])
                bounds_upper.extend([np.inf, t_max_fit, 150.0])

        # --- PERFORM GAUSSIAN FITTING ---
        fit_successful = False
        fitted_peaks = []
        total_fit_curve = np.zeros(len(clean_df))

        if enable_fit:
            try:
                popt, pcov = curve_fit(
                    multi_gaussian,
                    clean_df["Temperature"].values,
                    clean_df["TCD_Processed"].values,
                    p0=initial_guesses,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=5000
                )
                fit_successful = True
                total_fit_curve = multi_gaussian(clean_df["Temperature"].values, *popt)

                for p in range(num_peaks):
                    amp = popt[p*3]
                    center = popt[p*3+1]
                    sigma = popt[p*3+2]
                    peak_y = gaussian(clean_df["Temperature"].values, amp, center, sigma)
                    area = simpson(y=peak_y, x=clean_df["Temperature"].values)
                    fitted_peaks.append({
                        "Peak": f"Peak {p+1}",
                        "Center (°C)": center,
                        "Amplitude": amp,
                        "FWHM (°C)": 2.35482 * sigma,
                        "Area (a.u.*°C/g)": area,
                        "Curve": peak_y
                    })
            except Exception as fit_err:
                st.sidebar.warning(f"Gaussian Fit failed to converge: {fit_err}")

        # --- SIDEBAR: 3. GRAPHIC CONTROLS ---
        st.sidebar.header("3. Graph Customization")
        with st.sidebar.expander("✏️ Title & Typography Controls"):
            title_text = st.text_input("Plot Title", "CO₂-TPD Profile with Deconvoluted Peaks")
            title_bold = st.checkbox("Bold Title", value=True)
            title_italic = st.checkbox("Italic Title", value=False)
            title_size = st.slider("Title Size", 8, 24, 14)
            
            xlabel_text = st.text_input("X-Axis Label", "Temperature (°C)")
            xlabel_bold = st.checkbox("Bold X-Label", value=False)
            xlabel_italic = st.checkbox("Italic X-Label", value=False)
            
            ylabel_text = st.text_input("Y-Axis Label", "TCD Signal (a.u. / g_cat)")
            ylabel_bold = st.checkbox("Bold Y-Label", value=False)
            ylabel_italic = st.checkbox("Italic Y-Label", value=False)
            
            label_size = st.slider("Axis Label Size", 8, 20, 12)
            tick_size = st.slider("Tick Size", 6, 16, 10)
            font_family = st.selectbox("Font Style", ["DejaVu Sans", "DejaVu Serif", "Arial", "Times New Roman"])

        with st.sidebar.expander("🎨 Line Colors"):
            exp_color = st.color_picker("Experimental Data Line", "#000000")
            fit_color = st.color_picker("Cumulative Fit Line", "#FF0000")
            peak_colors = [st.color_picker(f"Peak {i+1} Fill Color", c) for i, c in enumerate(["#3182BD", "#E6550D", "#DE2D26", "#31A354"])]

        # --- PLOTTING ENGINE ---
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

        # Plot raw experimental data
        ax.plot(clean_df["Temperature"], clean_df["TCD_Processed"], color=exp_color, linewidth=1.5, label="Experimental")

        # Plot fitted peaks
        if fit_successful:
            ax.plot(clean_df["Temperature"], total_fit_curve, color=fit_color, linestyle="--", linewidth=1.5, label="Cumulative Fit")
            for i, p_info in enumerate(fitted_peaks):
                ax.plot(clean_df["Temperature"], p_info["Curve"], color=peak_colors[i], linewidth=1.2)
                ax.fill_between(clean_df["Temperature"], p_info["Curve"], color=peak_colors[i], alpha=0.3, label=f"{p_info['Peak']} ({p_info['Center (°C)']:.1f}°C)")

        def get_font_style(is_bold, is_italic):
            return ("bold" if is_bold else "normal"), ("italic" if is_italic else "normal")

        t_w, t_s = get_font_style(title_bold, title_italic)
        x_w, x_s = get_font_style(xlabel_bold, xlabel_italic)
        y_w, y_s = get_font_style(ylabel_bold, ylabel_italic)

        ax.set_title(title_text, fontsize=title_size, fontweight=t_w, fontstyle=t_s)
        ax.set_xlabel(xlabel_text, fontsize=label_size, fontweight=x_w, fontstyle=x_s)
        ax.set_ylabel(ylabel_text, fontsize=label_size, fontweight=y_w, fontstyle=y_s)
        ax.tick_params(axis="both", which="major", labelsize=tick_size)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=tick_size - 2, frameon=False)

        # --- DISPLAY & DASHBOARD ---
        col1, col2 = st.columns([1.6, 1])

        with col1:
            st.pyplot(fig)
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=600, bbox_inches="tight")

        with col2:
            st.subheader("📊 Deconvoluted Peak Results")
            if fit_successful:
                fit_df = pd.DataFrame(fitted_peaks)[["Peak", "Center (°C)", "FWHM (°C)", "Area (a.u.*°C/g)"]]
                total_fit_area = fit_df["Area (a.u.*°C/g)"].sum()
                fit_df["Fraction (%)"] = (fit_df["Area (a.u.*°C/g)"] / total_fit_area * 100).map("{:.1f} %".format)
                st.dataframe(fit_df, hide_index=True, use_container_width=True)
            else:
                st.info("Enable Gaussian fitting in the sidebar to view resolved peak parameters.")

    except Exception as e:
        st.error(f"Execution Error: {e}")
