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
    page_title="CO₂-TPD Advanced Analyzer & Benchmarker",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 CO₂-TPD Advanced Analyzer & Report Generator")
st.markdown(
    "Parse multi-block TPD files, customize publication graphics with rich typography, "
    "quantify basicity distribution, and export comprehensive reports in **PDF**, **Excel**, or **Word** formats."
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
    """Gaussian peak model."""
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def multi_gaussian(x, *params):
    """Sum of N Gaussian peaks."""
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        center = params[i + 1]
        sigma = params[i + 2]
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
        
        # Smart header detection
        default_header_idx = 0
        for r_idx, row_vals in raw_full.iterrows():
            row_str = " ".join([str(v) for v in row_vals.values if pd.notna(v)])
            if "Temperature" in row_str or "TCD" in row_str or "Signal" in row_str:
                default_header_idx = r_idx
                break

        header_row = st.sidebar.number_input("Header Row (0-indexed)", min_value=0, max_value=200, value=default_header_idx)
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

        def_temp_col = 0
        def_tcd_col = 1 if len(col_options) > 1 else 0

        # Auto-match column indices if keywords present
        for idx, opt in enumerate(col_options):
            if "temp" in opt.lower():
                def_temp_col = idx
            elif "tcd" in opt.lower() or "signal" in opt.lower():
                def_tcd_col = idx

        temp_col_sel = st.sidebar.selectbox("Temperature Column", col_options, index=def_temp_col)
        tcd_col_sel = st.sidebar.selectbox("TCD Signal Column", col_options, index=def_tcd_col)
        
        temp_idx = int(temp_col_sel.split("Idx ")[1].split(")")[0])
        tcd_idx = int(tcd_col_sel.split("Idx ")[1].split(")")[0])

        data_df = df_raw.iloc[header_row + 1:].copy()

        # Clean numeric data
        temp_data = clean_numeric_series(data_df.iloc[:, temp_idx])
        tcd_data = clean_numeric_series(data_df.iloc[:, tcd_idx])

        clean_df = pd.DataFrame({"Temperature": temp_data, "TCD": tcd_data}).dropna().sort_values("Temperature").reset_index(drop=True)

        if len(clean_df) < 5:
            st.error("Could not parse numeric data from selected columns. Please adjust the Header Row or Column selections.")
            st.stop()

        # --- SIDEBAR: 2. PROCESSING & TRIMMING ---
        st.sidebar.header("2. Signal Processing & Trimming")
        sample_mass = st.sidebar.number_input("Sample Mass (mg)", min_value=0.1, value=50.0, step=0.1)

        # Temperature Crop Sliders
        min_dataset_temp = float(np.floor(clean_df["Temperature"].min()))
        max_dataset_temp = float(np.ceil(clean_df["Temperature"].max()))

        temp_range = st.sidebar.slider(
            "Temperature Analysis Range (°C)",
            min_value=min_dataset_temp,
            max_value=max_dataset_temp,
            value=(min_dataset_temp, max_dataset_temp),
            step=5.0
        )

        clean_df = clean_df[(clean_df["Temperature"] >= temp_range[0]) & (clean_df["Temperature"] <= temp_range[1])].reset_index(drop=True)

        # Signal Normalization by Mass
        clean_df["TCD_Norm"] = clean_df["TCD"] / sample_mass

        # Baseline Correction
        baseline_mode = st.sidebar.selectbox("Baseline Subtraction", ["Linear (Start to End)", "Zero Offset (Min Shift)", "None"])
        if baseline_mode == "Linear (Start to End)":
            base_line = np.linspace(clean_df["TCD_Norm"].iloc[0], clean_df["TCD_Norm"].iloc[-1], len(clean_df))
            clean_df["TCD_Processed"] = (clean_df["TCD_Norm"] - base_line).clip(lower=0)
        elif baseline_mode == "Zero Offset (Min Shift)":
            clean_df["TCD_Processed"] = clean_df["TCD_Norm"] - clean_df["TCD_Norm"].min()
        else:
            clean_df["TCD_Processed"] = clean_df["TCD_Norm"]

        # --- SIDEBAR: 3. QUANTIFICATION & DECONVOLUTION ---
        st.sidebar.header("3. Basicity Quantification")
        analysis_mode = st.sidebar.radio("Analysis Method", ["Temperature Boundaries (Integration)", "Gaussian Peak Deconvolution"])

        fit_successful = False
        fitted_peaks = []
        total_fit_curve = np.zeros(len(clean_df))

        if analysis_mode == "Temperature Boundaries (Integration)":
            weak_boundary = st.sidebar.number_input("Weak / Medium Boundary (°C)", value=200.0, step=10.0)
            medium_boundary = st.sidebar.number_input("Medium / Strong Boundary (°C)", value=400.0, step=10.0)

            df_weak = clean_df[clean_df["Temperature"] < weak_boundary]
            df_med = clean_df[(clean_df["Temperature"] >= weak_boundary) & (clean_df["Temperature"] < medium_boundary)]
            df_strong = clean_df[clean_df["Temperature"] >= medium_boundary]

            area_weak = simpson(y=df_weak["TCD_Processed"], x=df_weak["Temperature"]) if len(df_weak) > 1 else 0.0
            area_med = simpson(y=df_med["TCD_Processed"], x=df_med["Temperature"]) if len(df_med) > 1 else 0.0
            area_strong = simpson(y=df_strong["TCD_Processed"], x=df_strong["Temperature"]) if len(df_strong) > 1 else 0.0
            total_area = area_weak + area_med + area_strong

        else:
            num_peaks = st.sidebar.slider("Number of Peaks to Fit", 1, 4, 3)
            initial_guesses = []
            bounds_lower = []
            bounds_upper = []

            default_centers = [150.0, 300.0, 500.0, 600.0]
            for p in range(num_peaks):
                with st.sidebar.expander(f"Peak {p+1} Guesses", expanded=False):
                    guess_amp = st.number_input(f"P{p+1} Amplitude", value=float(clean_df["TCD_Processed"].max() * 0.5), key=f"amp_{p}")
                    guess_center = st.number_input(f"P{p+1} Center (°C)", value=default_centers[min(p, 3)], key=f"cen_{p}")
                    guess_sigma = st.number_input(f"P{p+1} Sigma (°C)", value=30.0, key=f"sig_{p}")

                    initial_guesses.extend([guess_amp, guess_center, guess_sigma])
                    bounds_lower.extend([0.0, temp_range[0], 5.0])
                    bounds_upper.extend([np.inf, temp_range[1], 200.0])

            try:
                popt, _ = curve_fit(
                    multi_gaussian,
                    clean_df["Temperature"].values,
                    clean_df["TCD_Processed"].values,
                    p0=initial_guesses,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=5000
                )
                fit_successful = True
                total_fit_curve = multi_gaussian(clean_df["Temperature"].values, *popt)

                total_area = 0.0
                for p in range(num_peaks):
                    amp, center, sigma = popt[p*3], popt[p*3+1], popt[p*3+2]
                    peak_y = gaussian(clean_df["Temperature"].values, amp, center, sigma)
                    p_area = simpson(y=peak_y, x=clean_df["Temperature"].values)
                    total_area += p_area
                    fitted_peaks.append({
                        "Peak": f"Peak {p+1}",
                        "Center (°C)": center,
                        "FWHM (°C)": 2.35482 * sigma,
                        "Area": p_area,
                        "Curve": peak_y
                    })
            except Exception as e:
                st.sidebar.error(f"Fit did not converge: {e}")

        # --- SIDEBAR: 4. GRAPHICS CONTROLS ---
        st.sidebar.header("4. Typography & Plot Style")
        title_text = st.sidebar.text_input("Plot Title", "CO₂ Temperature-Programmed Desorption Profile")
        font_family = st.sidebar.selectbox("Font Family", ["DejaVu Sans", "DejaVu Serif", "Arial", "Times New Roman"])
        
        col_c1, col_c2 = st.sidebar.columns(2)
        line_color = col_c1.color_picker("Line Color", "#1F77B4")
        weak_color = col_c2.color_picker("Weak / Peak 1 Color", "#A6CEE3")
        med_color = col_c1.color_picker("Medium / Peak 2 Color", "#FDBF6F")
        strong_color = col_c2.color_picker("Strong / Peak 3 Color", "#FB9A99")

        # --- RENDER MAIN PLOT ---
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

        ax.plot(clean_df["Temperature"], clean_df["TCD_Processed"], color=line_color, linewidth=2, label="CO₂ Signal")

        if analysis_mode == "Temperature Boundaries (Integration)":
            if len(df_weak) > 0:
                ax.fill_between(df_weak["Temperature"], df_weak["TCD_Processed"], color=weak_color, alpha=0.5, label=f"Weak (<{weak_boundary:.0f}°C)")
            if len(df_med) > 0:
                ax.fill_between(df_med["Temperature"], df_med["TCD_Processed"], color=med_color, alpha=0.5, label=f"Medium ({weak_boundary:.0f}-{medium_boundary:.0f}°C)")
            if len(df_strong) > 0:
                ax.fill_between(df_strong["Temperature"], df_strong["TCD_Processed"], color=strong_color, alpha=0.5, label=f"Strong (>{medium_boundary:.0f}°C)")
        else:
            if fit_successful:
                ax.plot(clean_df["Temperature"], total_fit_curve, color="red", linestyle="--", linewidth=1.5, label="Cumulative Fit")
                colors_list = [weak_color, med_color, strong_color, "#CAB2D6"]
                for i, p_info in enumerate(fitted_peaks):
                    c = colors_list[i % len(colors_list)]
                    ax.plot(clean_df["Temperature"], p_info["Curve"], color=c, linewidth=1.2)
                    ax.fill_between(clean_df["Temperature"], p_info["Curve"], color=c, alpha=0.4, label=f"{p_info['Peak']} ({p_info['Center (°C)']:.1f}°C)")

        ax.set_title(title_text, fontsize=13, fontweight="bold")
        ax.set_xlabel("Temperature (°C)", fontsize=11)
        ax.set_ylabel("TCD Signal (a.u. / g_cat)", fontsize=11)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=9, frameon=False)

        # --- DASHBOARD LAYOUT ---
        col_plot, col_results = st.columns([1.6, 1])

        with col_plot:
            st.pyplot(fig)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button("📥 Download High-Res Plot (PNG)", data=img_buf.getvalue(), file_name="CO2_TPD_Plot.png", mime="image/png")

        with col_results:
            st.subheader("📊 Quantified Basic Sites")

            if analysis_mode == "Temperature Boundaries (Integration)":
                res_data = [
                    {"Site Type": "Weak Sites", "Temp Range": f"< {weak_boundary:.0f} °C", "Desorption Area": round(area_weak, 2), "Fraction (%)": f"{(area_weak/total_area*100 if total_area else 0):.1f} %"},
                    {"Site Type": "Medium Sites", "Temp Range": f"{weak_boundary:.0f} – {medium_boundary:.0f} °C", "Desorption Area": round(area_med, 2), "Fraction (%)": f"{(area_med/total_area*100 if total_area else 0):.1f} %"},
                    {"Site Type": "Strong Sites", "Temp Range": f"> {medium_boundary:.0f} °C", "Desorption Area": round(area_strong, 2), "Fraction (%)": f"{(area_strong/total_area*100 if total_area else 0):.1f} %"},
                    {"Site Type": "Total Basicity", "Temp Range": "Full Spectrum", "Desorption Area": round(total_area, 2), "Fraction (%)": "100.0 %"}
                ]
                res_df = pd.DataFrame(res_data)
                st.dataframe(res_df, hide_index=True, use_container_width=True)

            else:
                if fit_successful:
                    res_list = []
                    for p in fitted_peaks:
                        frac = (p["Area"] / total_area * 100) if total_area > 0 else 0.0
                        res_list.append({
                            "Peak": p["Peak"],
                            "Center (°C)": round(p["Center (°C)"], 1),
                            "FWHM (°C)": round(p["FWHM (°C)"], 1),
                            "Area": round(p["Area"], 2),
                            "Fraction (%)": f"{frac:.1f} %"
                        })
                    res_df = pd.DataFrame(res_list)
                    st.dataframe(res_df, hide_index=True, use_container_width=True)

            max_idx = clean_df["TCD_Processed"].idxmax()
            primary_tmax = clean_df.loc[max_idx, "Temperature"]
            st.metric("Primary Peak Temp (T_max)", f"{primary_tmax:.1f} °C")
            st.metric("Sample Mass Evaluated", f"{sample_mass:.1f} mg")

    except Exception as err:
        st.error(f"Error parsing file or rendering profile: {err}")
