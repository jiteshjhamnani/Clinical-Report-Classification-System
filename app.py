from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="Clinical Report Classification System",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000/predict"


def load_text_file(filename: str) -> str:
    return (BASE_DIR / filename).read_text(encoding="utf-8")


def inject_css() -> None:
    css = load_text_file("styles.css")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_hero():
    hero_html = load_text_file("hero.html")
    st.markdown(hero_html, unsafe_allow_html=True)


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n".join(chunks).strip()


def build_download_text(report: str, result: dict) -> str:
    lines = [
        "Clinical Report Analysis Report",
        "=" * 34,
        "",
        f"Predicted Specialty: {result.get('specialty', '-')}",
        f"Confidence: {result.get('confidence', '-')}",
        f"Readability Score: {result.get('readability_score', '-')}",
        f"Complexity: {result.get('complexity', '-')}",
        f"Medical Score: {result.get('medical_score', '-')}",
        "",
        "Top Predictions:",
    ]

    for i, pred in enumerate(result.get("top_predictions", []), start=1):
        lines.append(
            f"{i}. {pred.get('specialty', '-')} ({pred.get('confidence', '-')}%)"
        )

    lines.extend([
        "",
        "Original Report:",
        "-" * 16,
        report
    ])

    return "\n".join(lines)


def metric_card(icon: str, label: str, value: str, subtext: str, tone: str = "blue") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-ico {tone}">{icon}</div>
            <div class="metric-label">{label}</div>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """


def render_result(result: dict, report: str) -> None:
    st.markdown(
        """
        <div class="analysis-header">
            <div>
                <div class="analysis-title">Your report has been analyzed</div>
                <div class="analysis-subtitle">
                    Here are the extracted insights from your medical report.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    top_bar_cols = st.columns([1.8, 0.9, 0.9])

    download_text = build_download_text(report, result)

    with top_bar_cols[1]:
        st.download_button(
            "⬇ Download Report",
            data=download_text,
            file_name="clinical_report_analysis.txt",
            mime="text/plain",
            use_container_width=True
        )

    with top_bar_cols[2]:
        if st.button("🔖 Save Report", use_container_width=True):
            saved = st.session_state.get("saved_reports", [])
            saved.append(result)
            st.session_state.saved_reports = saved
            st.success("Report saved in this session.")

    left_col, right_col = st.columns([1.15, 1])

    with left_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="specialty-layout">
                    <div class="big-icon">🏥</div>
                    <div>
                        <div class="small-label">Predicted Specialty</div>
                        <div class="specialty-name">{result.get('specialty', '-')}</div>
                        <div class="confidence-badge">Confidence: {result.get('confidence', '-')}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right_col:
        st.markdown(
            """
            <div class="pred-box">
                <div class="pred-title">Top Predictions</div>
            """,
            unsafe_allow_html=True
        )

        for idx, pred in enumerate(result.get("top_predictions", []), start=1):
            st.markdown(
                f"""
                <div class="pred-row">
                    <div class="pred-rank">{idx}</div>
                    <div class="pred-name">{pred.get('specialty', '-')}</div>
                    <div class="pred-score">{pred.get('confidence', '-')}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.markdown(
            metric_card("📄", "Readability Score", str(result.get("readability_score", "-")),
                        "Flesch-Kincaid grade", tone="blue"),
            unsafe_allow_html=True
        )

    with metric_cols[1]:
        st.markdown(
            metric_card("🧠", "Complexity", str(result.get("complexity", "-")),
                        "Text complexity level", tone="amber"),
            unsafe_allow_html=True
        )

    with metric_cols[2]:
        st.markdown(
            metric_card("🧪", "Medical Score", str(result.get("medical_score", "-")),
                        "Medical content density", tone="green"),
            unsafe_allow_html=True
        )

    with metric_cols[3]:
        st.markdown(
            metric_card("🎯", "Confidence", f"{result.get('confidence', '-')}%",
                        "Model confidence score", tone="cyan"),
            unsafe_allow_html=True
        )

    st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([1.6, 1])

    with bottom_left:
        specialty     = result.get("specialty", "the predicted specialty")
        readability   = result.get("readability_score", "-")
        complexity    = result.get("complexity", "-")
        medical_score = result.get("medical_score", "-")

        st.markdown(
            f"""
            <div class="info-card">
                <h3>What this means</h3>
                <p>
                    The report is most likely from the <b>{specialty}</b> department.
                    It has a readability score of <b>{readability}</b> and a complexity
                    level of <b>{complexity}</b>. The medical content density score is
                    <b>{medical_score}</b>, which shows how much domain-specific
                    terminology appears in the report.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with bottom_right:
        st.markdown(
            """
            <div class="tips-card">
                <h3>Tips</h3>
                <ul>
                    <li>Readability score helps you understand how easy the report is to read.</li>
                    <li>Higher medical score usually means more domain-specific content.</li>
                    <li>Confidence shows how certain the model is about the predicted specialty.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Prediction Distribution</div>', unsafe_allow_html=True)

    pred_df = pd.DataFrame(result.get("top_predictions", []))
    if not pred_df.empty:
        st.bar_chart(pred_df.set_index("specialty")["confidence"])

    with st.expander("View Raw API Response"):
        st.json(result)


# ---------------------------
# SESSION STATE
# ---------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "upload"

if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if "saved_reports" not in st.session_state:
    st.session_state.saved_reports = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ---------------------------
# RENDER PAGE
# ---------------------------
inject_css()
render_hero()

# ── Mode selector — single heading, no duplicate ──
st.markdown('<div class="section-title">Choose how you want to add the report</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("📄 Upload PDF Report", use_container_width=True):
        st.session_state.mode = "upload"

with col2:
    if st.button("✏️ Enter Manually", use_container_width=True):
        st.session_state.mode = "manual"

st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

# ── Input area ──
if st.session_state.mode == "upload":
    st.markdown(
        '<p style="color:#475569; font-size:14px; font-weight:600; margin-bottom:6px;">📎 Upload your PDF report below:</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop your PDF here or click Browse files",
        type=["pdf"],
        label_visibility="visible"
    )

    st.markdown(
        '<div class="helper-text">Supports PDF reports. If the file is scanned image-only, text extraction may be limited.</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:
        try:
            extracted_text = extract_text_from_pdf(uploaded_file)
            if extracted_text.strip():
                st.session_state.report_text = extracted_text
                st.success("✅ Text extracted from PDF successfully.")
            else:
                st.warning("⚠️ No text could be extracted from this PDF.")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

else:
    st.markdown(
        '<p style="color:#475569; font-size:14px; font-weight:600; margin-bottom:6px;">✏️ Paste your clinical report below:</p>',
        unsafe_allow_html=True
    )

report = st.text_area(
    "Medical Report Text",
    height=280,
    value=st.session_state.report_text,
    placeholder="Paste the clinical report here...",
    label_visibility="collapsed"
)

st.session_state.report_text = report

st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
analyze_clicked = st.button("🚀 Analyze Report", use_container_width=True)


if analyze_clicked:
    if not report.strip():
        st.warning("⚠️ Please enter or upload a medical report first.")
    else:
        with st.spinner("Analyzing report..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"report": report},
                    timeout=120
                )
                if response.status_code != 200:
                    st.error(f"API error ({response.status_code}): {response.text}")
                    st.stop()
                result = response.json()
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
                st.stop()

        if result.get("status") == "rejected":
            st.error(result.get("message", "Input rejected."))
            st.stop()

        st.session_state.last_result = result
        render_result(result, report)

elif st.session_state.last_result:
    st.markdown('<div class="section-title">Last Analysis</div>', unsafe_allow_html=True)
    render_result(st.session_state.last_result, st.session_state.report_text)