import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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
st.caption(
    "⚠️ This tool supports interpretation, but it does not replace complementary characterization "
    "(DRIFTS/IR for carbonate speciation, calibrated pulses for absolute quantification). "
    "Treat the auto-generated 'literature correlation' text as a hypothesis to check, not a verdict."
)


# ----------------------------- HELPERS ----------------------------- #

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


def gaussian(x, amp, cen, wid):
    wid = max(wid, 1e-6)
    return amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)


def multi_gaussian(x, *params):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        y += gaussian(x, params[i], params[i + 1], params[i + 2])
    return y


def safe_font(requested_family):
    """Fall back to DejaVu Sans if the requested font isn't actually installed,
    instead of silently mis-rendering or erroring."""
    available = {f.name for f in fontManager.ttflist}
    return requested_family if requested_family in available else "DejaVu Sans"


def fit_gaussian_peaks(temp, signal, prominence_frac=0.05, max_peaks=6,
                        min_separation_frac=0.03, min_relative_area=0.05):
    """Detect peaks and fit a sum-of-Gaussians model. Returns list of dicts
    with center, amplitude, width, and analytic area, or None if fitting fails.

    Guards against overfitting noise as real peaks:
    - `min_separation_frac` enforces a minimum spacing between detected peaks
      (as a fraction of the temperature span) so noise bumps near a real peak
      aren't picked up as separate features.
    - After fitting, any component whose area is below `min_relative_area`
      of the largest fitted peak's area is dropped as noise, and the
      remaining peaks are re-fit for cleaner parameters.
    """
    if len(temp) < 10 or np.nanmax(signal) <= 0:
        return None

    # Estimate the local noise floor (std of signal minus a coarse moving-average
    # trend) so flat/noisy baselines with no real desorption feature don't get
    # noise bumps fitted as fake peaks.
    win = max(5, len(signal) // 20)
    kernel = np.ones(win) / win
    smoothed = np.convolve(signal, kernel, mode="same")
    noise_std = np.std(signal - smoothed)
    if noise_std == 0:
        noise_std = 1e-9

    span = temp.max() - temp.min()
    min_dist_idx = max(1, int(len(temp) * min_separation_frac))
    prominence = max(prominence_frac * np.nanmax(signal), 6 * noise_std)
    peak_idx, _ = find_peaks(signal, prominence=prominence, distance=min_dist_idx)
    if len(peak_idx) == 0:
        return None
    if len(peak_idx) > max_peaks:
        order = np.argsort(signal[peak_idx])[::-1][:max_peaks]
        peak_idx = np.sort(peak_idx[order])

    def build_and_fit(idx_list):
        init_params, lower_bounds, upper_bounds = [], [], []
        for idx in idx_list:
            amp0 = signal[idx]
            cen0 = temp[idx]
            wid0 = max(span / (3 * len(idx_list)), 5.0)
            init_params += [amp0, cen0, wid0]
            lower_bounds += [0.0, temp.min() - 10, 1.0]
            upper_bounds += [amp0 * 3 + 1e-9, temp.max() + 10, span]
        popt, _ = curve_fit(
            multi_gaussian, temp, signal,
            p0=init_params, bounds=(lower_bounds, upper_bounds),
            maxfev=20000
        )
        peaks = []
        for i in range(0, len(popt), 3):
            amp, cen, wid = popt[i], popt[i + 1], popt[i + 2]
            area = amp * abs(wid) * np.sqrt(2 * np.pi)
            peaks.append({"amplitude": amp, "center": cen, "width": wid, "area": area})
        return peaks

    try:
        peaks = build_and_fit(peak_idx)
    except Exception:
        return None

    # Drop components that are negligible relative to the largest peak (fit noise),
    # then re-fit with just the surviving peak centers for cleaner parameters.
    if peaks:
        max_area = max(p["area"] for p in peaks)
        survivors_idx = [idx for idx, p in zip(peak_idx, peaks) if p["area"] >= min_relative_area * max_area]
        if survivors_idx and len(survivors_idx) != len(peak_idx):
            try:
                peaks = build_and_fit(survivors_idx)
            except Exception:
                pass  # keep the original fit if the re-fit fails

    peaks.sort(key=lambda p: p["center"])
    return peaks


# --------------------- SIDEBAR: 1. DATA INPUT ----------------------- #
st.sidebar.header("1. Data Input & Mapping")
uploaded_file = st.sidebar.file_uploader("Upload TPD Excel (.xlsx / .xls)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()
        engine = "xlrd" if filename.endswith(".xls") else "openpyxl"

        excel_file = pd.ExcelFile(uploaded_file, engine=engine)
        sheet_name = st.sidebar.selectbox("Select Excel Sheet", excel_file.sheet_names)

        raw_full = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, engine=engine)

        default_header_idx = min(23, max(len(raw_full) - 1, 0))
        for r_idx, row_vals in raw_full.iterrows():
            row_str = " ".join([str(v) for v in row_vals.values if pd.notna(v)])
            if "Temperature" in row_str and "TCD" in row_str:
                default_header_idx = min(r_idx + 3, max(len(raw_full) - 1, 0))
                break

        header_row = st.sidebar.number_input(
            "Header Row (0-indexed)", min_value=0,
            max_value=max(len(raw_full) - 1, 0), value=default_header_idx
        )
        df_raw = raw_full  # already read once above; no need to re-read the file

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
        def_tcd_col = 13 if len(col_options) > 13 else min(1, len(col_options) - 1)

        temp_col_sel = st.sidebar.selectbox("Temperature Column", col_options, index=def_temp_col)
        tcd_col_sel = st.sidebar.selectbox("TCD Signal Column", col_options, index=def_tcd_col)

        temp_idx = int(temp_col_sel.split("Idx ")[1].split(")")[0])
        tcd_idx = int(tcd_col_sel.split("Idx ")[1].split(")")[0])

        if temp_idx == tcd_idx:
            st.sidebar.warning("⚠️ Temperature and TCD Signal are mapped to the same column — check your selection.")

        data_df = df_raw.iloc[header_row + 1:].copy()

        # --------------------- SIDEBAR: SAMPLE CONTEXT ----------------------- #
        st.sidebar.header("2. Sample & Pretreatment Context")
        catalyst_state = st.sidebar.radio(
            "Catalyst state analyzed",
            ["Reduced (post H2-TPR, as used in catalysis)", "Oxidized / calcined (as synthesized)"],
            index=0,
            help=("If this catalyst is used in reduced form during CO2 hydrogenation, CO2-TPD should be run "
                  "directly after H2-TPR (cooled under inert gas, no re-exposure to air) so the basic sites "
                  "measured reflect the actual working surface.")
        )
        sample_mass = st.sidebar.number_input("Sample Mass (mg)", min_value=0.1, value=50.0, step=0.1)

        st.sidebar.markdown("**Baseline correction window** (pick flat regions before/after desorption)")
        baseline_subtraction = st.sidebar.checkbox("Apply Linear Baseline Correction", value=True)
        baseline_mode = st.sidebar.radio(
            "Baseline reference",
            ["Use first/last data point (simple)", "Use averaged start/end windows (more robust)"],
            index=1
        )
        if baseline_mode == "Use averaged start/end windows (more robust)":
            bwin = st.sidebar.slider("Baseline averaging window width (°C)", 5, 100, 20, step=5)

        calibration_factor = st.sidebar.number_input(
            "CO2 Calibration Factor (µmol CO₂ per unit peak area). Leave at 0 to report relative area only.",
            min_value=0.0, value=0.0, step=0.01, format="%.4f",
            help="Only fill this in if you ran a calibrated CO2 pulse loop of known volume/concentration. "
                 "Without it, 'quantified' values below are relative peak areas (a.u.), not absolute µmol/g."
        )

        # ------------------ SIDEBAR: BASIC SITE BOUNDARIES ------------------- #
        st.sidebar.header("3. Basic Site Temperature Boundaries")
        st.sidebar.caption(
            "These are conventional cutoffs, not physical boundaries — real peaks overlap. "
            "Use the Gaussian peak-fitting option below for a less arbitrary split."
        )
        weak_max = st.sidebar.number_input("Weak / Medium Boundary (°C)", min_value=50.0, max_value=300.0, value=200.0, step=10.0)
        medium_max = st.sidebar.number_input("Medium / Strong Boundary (°C)", min_value=200.0, max_value=600.0, value=400.0, step=10.0)

        use_deconvolution = st.sidebar.checkbox(
            "Also run Gaussian peak deconvolution (recommended)", value=True,
            help="Fits overlapping desorption peaks individually instead of splitting at fixed temperatures."
        )

        # --------------------------- PRE-PROCESSING --------------------------- #
        clean_df = pd.DataFrame({
            "Temperature": clean_numeric_series(data_df.iloc[:, temp_idx]),
            "TCD": clean_numeric_series(data_df.iloc[:, tcd_idx])
        }).dropna()

        # collapse duplicate temperature readings (keeps Simpson integration well-defined)
        clean_df = clean_df.groupby("Temperature", as_index=False)["TCD"].mean()
        clean_df = clean_df.sort_values("Temperature").reset_index(drop=True)

        if len(clean_df) < 5:
            st.error("Not enough numeric data points found. Adjust 'Header Row' or check column mapping.")
            st.stop()

        # Note: this is a mass-normalized signal (a.u. per g_cat), not an absolute concentration
        # unless a calibration factor is supplied above.
        clean_df["TCD_Norm"] = (clean_df["TCD"] / sample_mass) * 1000

        if baseline_subtraction:
            if baseline_mode == "Use first/last data point (simple)":
                start_val = clean_df["TCD_Norm"].iloc[0]
                end_val = clean_df["TCD_Norm"].iloc[-1]
            else:
                start_mask = clean_df["Temperature"] <= (clean_df["Temperature"].min() + bwin)
                end_mask = clean_df["Temperature"] >= (clean_df["Temperature"].max() - bwin)
                start_val = clean_df.loc[start_mask, "TCD_Norm"].mean()
                end_val = clean_df.loc[end_mask, "TCD_Norm"].mean()
                if pd.isna(start_val):
                    start_val = clean_df["TCD_Norm"].iloc[0]
                if pd.isna(end_val):
                    end_val = clean_df["TCD_Norm"].iloc[-1]
            baseline = np.linspace(start_val, end_val, len(clean_df))
            clean_df["TCD_Processed"] = (clean_df["TCD_Norm"] - baseline).clip(lower=0)
        else:
            clean_df["TCD_Processed"] = clean_df["TCD_Norm"]

        temp_arr = clean_df["Temperature"].values
        signal_arr = clean_df["TCD_Processed"].values

        # ---------------------------- REGIONAL INTEGRATION (binned) ---------------------------- #
        total_area = simpson(y=signal_arr, x=temp_arr)

        df_weak = clean_df[clean_df["Temperature"] < weak_max]
        df_med = clean_df[(clean_df["Temperature"] >= weak_max) & (clean_df["Temperature"] < medium_max)]
        df_strong = clean_df[clean_df["Temperature"] >= medium_max]

        area_weak = simpson(y=df_weak["TCD_Processed"].values, x=df_weak["Temperature"].values) if len(df_weak) > 1 else 0.0
        area_med = simpson(y=df_med["TCD_Processed"].values, x=df_med["Temperature"].values) if len(df_med) > 1 else 0.0
        area_strong = simpson(y=df_strong["TCD_Processed"].values, x=df_strong["Temperature"].values) if len(df_strong) > 1 else 0.0

        pct_weak = (area_weak / total_area * 100) if total_area > 0 else 0
        pct_med = (area_med / total_area * 100) if total_area > 0 else 0
        pct_strong = (area_strong / total_area * 100) if total_area > 0 else 0

        # ---------------------------- GAUSSIAN DECONVOLUTION ---------------------------- #
        fitted_peaks = fit_gaussian_peaks(temp_arr, signal_arr) if use_deconvolution else None

        if fitted_peaks:
            deconv_total_area = sum(p["area"] for p in fitted_peaks)
            deconv_weak = sum(p["area"] for p in fitted_peaks if p["center"] < weak_max)
            deconv_med = sum(p["area"] for p in fitted_peaks if weak_max <= p["center"] < medium_max)
            deconv_strong = sum(p["area"] for p in fitted_peaks if p["center"] >= medium_max)
            deconv_pct_weak = deconv_weak / deconv_total_area * 100 if deconv_total_area > 0 else 0
            deconv_pct_med = deconv_med / deconv_total_area * 100 if deconv_total_area > 0 else 0
            deconv_pct_strong = deconv_strong / deconv_total_area * 100 if deconv_total_area > 0 else 0

        # --------------------- SIDEBAR: 4. GRAPHIC & AXIS CONTROLS -------------------- #
        st.sidebar.header("4. Graph Customization & Axes")

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
            show_fit_overlay = st.checkbox("Overlay fitted Gaussian peaks (if computed)", value=True)

        with st.sidebar.expander("📐 Scale Ranges & Rescaling", expanded=False):
            auto_x = st.checkbox("Auto X-Axis Range", value=True)
            min_temp, max_temp = float(clean_df["Temperature"].min()), float(clean_df["Temperature"].max())
            if not auto_x:
                x_min = st.number_input("X Min (°C)", value=min_temp, step=10.0)
                x_max = st.number_input("X Max (°C)", value=max_temp, step=10.0)

            auto_y = st.checkbox("Auto Y-Axis Range", value=True)
            max_tcd = float(clean_df["TCD_Processed"].max())
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

        # ------------------------------ PLOTTING ------------------------------ #
        plt.rcParams["font.family"] = safe_font(font_family)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

        ax.plot(clean_df["Temperature"], clean_df["TCD_Processed"], color=curve_color, linewidth=line_width, label="CO₂ Signal")

        if color_mode == "Shade by Basic Site Regions":
            ax.fill_between(df_weak["Temperature"], df_weak["TCD_Processed"], color=weak_color, alpha=fill_alpha, label=f"Weak (<{weak_max:.0f}°C)")
            ax.fill_between(df_med["Temperature"], df_med["TCD_Processed"], color=med_color, alpha=fill_alpha, label=f"Medium ({weak_max:.0f}-{medium_max:.0f}°C)")
            ax.fill_between(df_strong["Temperature"], df_strong["TCD_Processed"], color=strong_color, alpha=fill_alpha, label=f"Strong (>{medium_max:.0f}°C)")
        else:
            ax.fill_between(clean_df["Temperature"], clean_df["TCD_Processed"], color=single_fill_color, alpha=fill_alpha)

        if fitted_peaks and show_fit_overlay:
            x_dense = np.linspace(temp_arr.min(), temp_arr.max(), 1000)
            for p in fitted_peaks:
                ax.plot(x_dense, gaussian(x_dense, p["amplitude"], p["center"], p["width"]),
                        linestyle="--", linewidth=max(line_width * 0.7, 0.8), color="black", alpha=0.6)
            fit_sum = multi_gaussian(x_dense, *[v for p in fitted_peaks for v in (p["amplitude"], p["center"], p["width"])])
            ax.plot(x_dense, fit_sum, linestyle=":", linewidth=line_width, color="dimgray", label="Sum of fitted peaks")

        if color_mode == "Shade by Basic Site Regions" or (fitted_peaks and show_fit_overlay):
            ax.legend(fontsize=max(tick_size - 1, 6), frameon=False)

        ax.set_xlabel(xlabel_text, fontsize=label_size, fontweight=font_weight)
        ax.set_ylabel(ylabel_text, fontsize=label_size, fontweight=font_weight)
        ax.set_title(title_text, fontsize=title_size, fontweight=font_weight)

        if not auto_x:
            ax.set_xlim(x_min, x_max)
        if not auto_y:
            ax.set_ylim(y_min, y_max)

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

        # ------------------------------ MAIN DISPLAY ------------------------------ #
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
            unit_label = "µmol CO₂/g_cat" if calibration_factor > 0 else "a.u. (relative)"
            st.subheader("📊 Simple Binned Quantification")
            st.caption(f"Fixed cutoffs at {weak_max:.0f}°C / {medium_max:.0f}°C. Units: {unit_label}.")

            def conv(a):
                return a * calibration_factor if calibration_factor > 0 else a

            summary_table = pd.DataFrame({
                "Site Type": ["Weak Sites", "Medium Sites", "Strong Sites", "Total Surface Basicity"],
                "Temp Range": [f"< {weak_max:.0f} °C", f"{weak_max:.0f} – {medium_max:.0f} °C", f"> {medium_max:.0f} °C", "Full Profile"],
                f"Area ({unit_label})": [f"{conv(area_weak):.2f}", f"{conv(area_med):.2f}", f"{conv(area_strong):.2f}", f"{conv(total_area):.2f}"],
                "Fraction (%)": [f"{pct_weak:.1f} %", f"{pct_med:.1f} %", f"{pct_strong:.1f} %", "100.0 %"]
            })
            st.dataframe(summary_table, hide_index=True, use_container_width=True)

            if fitted_peaks:
                st.subheader("🎯 Gaussian Peak-Fitted Quantification")
                st.caption("Each detected peak fitted individually, then grouped by fitted center temperature. "
                           "More robust to overlap than the fixed-cutoff table above.")
                peak_rows = []
                for i, p in enumerate(fitted_peaks, start=1):
                    peak_rows.append({
                        "Peak #": i,
                        "T_center (°C)": f"{p['center']:.1f}",
                        "Width (σ, °C)": f"{p['width']:.1f}",
                        f"Area ({unit_label})": f"{conv(p['area']):.2f}",
                        "Fraction (%)": f"{(p['area']/deconv_total_area*100 if deconv_total_area>0 else 0):.1f} %"
                    })
                st.dataframe(pd.DataFrame(peak_rows), hide_index=True, use_container_width=True)
                st.write(
                    f"Grouped by fitted center → Weak: **{deconv_pct_weak:.1f}%**, "
                    f"Medium: **{deconv_pct_med:.1f}%**, Strong: **{deconv_pct_strong:.1f}%**"
                )
            elif use_deconvolution:
                st.info("Peak fitting did not converge on distinguishable peaks for this profile/settings — "
                         "showing binned quantification only. Try adjusting data quality or use the binned table above.")

            max_idx = clean_df["TCD_Processed"].idxmax()
            peak_temp = clean_df.loc[max_idx, "Temperature"]
            st.metric("Primary Peak Temperature (T_max)", f"{peak_temp:.1f} °C")
            st.metric("Sample Mass Evaluated", f"{sample_mass} mg")
            if calibration_factor == 0:
                st.caption("ℹ️ No calibration factor entered — areas above are relative (a.u.), not absolute µmol/g.")

        # --------------------- LITERATURE COMPARISON MODULE --------------------- #
        st.markdown("---")
        st.header("🧠 Literature Comparison & Methanol Synthesis Correlation")
        st.caption(
            "These statements describe general trends reported for related catalyst families "
            "(e.g. Cu/ZnO/ZrO₂, In₂O₃-based systems) under CO2/CO/H2 hydrogenation. They are not a "
            "confirmed characterization of your specific sample — carbonate speciation in particular "
            "requires DRIFTS/IR, not TPD alone."
        )

        # use the deconvolution fractions when available (more defensible), else fall back to binned
        eval_pct_weak = deconv_pct_weak if fitted_peaks else pct_weak
        eval_pct_med = deconv_pct_med if fitted_peaks else pct_med
        eval_pct_strong = deconv_pct_strong if fitted_peaks else pct_strong

        cat_col1, cat_col2 = st.columns(2)

        with cat_col1:
            st.subheader("📌 Main Observations from Profile")

            dominant_site = "Weak" if eval_pct_weak >= max(eval_pct_med, eval_pct_strong) else \
                ("Medium" if eval_pct_med >= eval_pct_strong else "Strong")

            st.write(f"• **Dominant Surface Species (by fitted area):** {dominant_site} basic-site region "
                     f"({max(eval_pct_weak, eval_pct_med, eval_pct_strong):.1f}% of total basicity).")
            st.write(f"• **Desorption Thermal Maxima ($T_{{max}}$):** Primary peak occurs at **{peak_temp:.1f} °C**.")
            st.write(f"• **Catalyst state analyzed:** {catalyst_state}")

            if peak_temp < 200:
                st.write("• **Possible site nature:** commonly associated with weak hydroxyl groups "
                         "($\\text{OH}^-$) or loosely bound bicarbonate-like species — consistent with, "
                         "but not confirmed by, TPD alone.")
            elif 200 <= peak_temp <= 400:
                st.write("• **Possible site nature:** commonly associated with medium-strength metal–oxygen "
                         "pairs (e.g. Cu–O–Zr, Zn–O–Zr) or oxygen vacancies coordinating bidentate carbonate-type "
                         "species — carbonate speciation would need DRIFTS/IR to confirm.")
            elif peak_temp <= 550:
                st.write("• **Possible site nature:** commonly associated with strong, isolated low-coordinated "
                         "oxygen anions forming more strongly bound carbonate species — again, speciation is "
                         "inferred, not confirmed, by TPD alone.")
            else:
                st.warning(
                    "⚠️ The primary peak is above ~550 °C. At this temperature, signal is often dominated by "
                    "**decomposition of bulk-like/polydentate carbonates or the support itself** (e.g. residual "
                    "carbonate on ZrO₂/ZnO), rather than classical 'very strong basic sites'. Consider TGA-MS or "
                    "checking whether this feature persists after longer support calcination before interpreting "
                    "it as basicity."
                )

        with cat_col2:
            st.subheader("💡 Correlation to $\\text{CO}_2/\\text{CO}/\\text{H}_2$ Methanol Synthesis")

            if eval_pct_med >= 40.0:
                st.success("🟢 **Profile consistent with good CO2-activation potential**")
                st.write(
                    "**Literature trend:** in $\\text{CO}_2/\\text{CO}/\\text{H}_2$ feeds over systems like "
                    "$\\text{Cu/ZnO/ZrO}_2$ or $\\text{In}_2\\text{O}_3$, medium-strength basic sites "
                    "($200\\text{–}400\\,^\\circ\\text{C}$) are often reported as favorable for $\\text{CO}_2$ "
                    "activation, binding it with enough strength to stabilize $*\\text{HCOO}$-type intermediates "
                    "without over-poisoning the surface. Whether that holds for *this* catalyst still depends on "
                    "metal dispersion, promoter effects, and reaction conditions — this is a favorable *indicator*, "
                    "not a guarantee of performance."
                )
            elif eval_pct_weak > 50.0:
                st.warning("🟡 **Profile suggests weak-binding-dominated CO2 adsorption**")
                st.write(
                    "**Literature trend:** a high share of weak basic sites ($<200\\,^\\circ\\text{C}$) is often "
                    "associated with short $\\text{CO}_2$ residence time and desorption before hydrogenation can "
                    "occur, which *may* correlate with lower single-pass conversion around typical methanol-"
                    "synthesis conditions ($220\\text{–}260\\,^\\circ\\text{C}$). Confirm with actual activity testing."
                )
            else:
                st.error("🔴 **Profile suggests possible strong-site-dominated CO2 retention**")
                st.write(
                    "**Literature trend:** a high share of strong basic sites ($>400\\,^\\circ\\text{C}$) is "
                    "sometimes linked to excessively strong $\\text{CO}_2$ binding, which in some systems is "
                    "associated with reduced methanol selectivity in favor of deep hydrogenation to "
                    "$\\text{CH}_4$ or RWGS to $\\text{CO}$. This is a plausible hypothesis to test with activity "
                    "data, not a confirmed mechanism for your sample."
                )

    except Exception as e:
        st.error(f"⚠️ Error executing analysis: {e}")
        st.info("Common causes: wrong Header Row, Temperature/TCD columns mapped incorrectly, "
                 "or a sheet that doesn't contain a clean numeric TPD block. Adjust the sidebar settings and retry.")
else:
    st.info("👋 Upload a CO₂-TPD dataset via the sidebar to access custom plotting, quantification, and literature comparison.")
