from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from streamlit.errors import StreamlitSecretNotFoundError
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except StreamlitSecretNotFoundError:
        groq_api_key = None

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    reasoning_format="parsed",
    api_key=groq_api_key,
) if groq_api_key else None

st.set_page_config(page_title="Blood Work Analyzer", layout="wide")

st.markdown("""
<style>
.st-key-health-output,
.st-key-diet-output {
    box-sizing: border-box;
    max-width: 100%;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    color: var(--text-color);
    font-size: 0.95rem;
    line-height: 1.65;
    overflow-wrap: anywhere;
}
.st-key-health-output {
    background: color-mix(in srgb, var(--primary-color) 7%, var(--secondary-background-color));
}
.st-key-diet-output {
    background: color-mix(in srgb, var(--primary-color) 12%, var(--secondary-background-color));
}
.st-key-health-output p,
.st-key-health-output li,
.st-key-diet-output p,
.st-key-diet-output li,
.st-key-health-output h1,
.st-key-health-output h2,
.st-key-health-output h3,
.st-key-diet-output h1,
.st-key-diet-output h2,
.st-key-diet-output h3 {
    color: var(--text-color);
}
.st-key-health-output h1,
.st-key-health-output h2,
.st-key-health-output h3,
.st-key-diet-output h1,
.st-key-diet-output h2,
.st-key-diet-output h3 {
    line-height: 1.3;
}
.st-key-health-output strong,
.st-key-diet-output strong {
    color: var(--text-color);
    font-weight: 700;
}
@media (max-width: 640px) {
    .st-key-health-output,
    .st-key-diet-output {
        font-size: 0.9rem;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("Blood Work Analyzer")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Blood Work Report")
    blood_report = st.text_area(
        label="Paste your report below",
        height=500,
        placeholder="Paste your blood work report here...",
        label_visibility="collapsed"
    )
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

with right_col:
    st.subheader("Health Summary")
    health_box = st.container(height=230, border=True, key="health-output")

    st.subheader("Suggested Diet Plan")
    diet_box = st.container(height=230, border=True, key="diet-output")

if analyze_clicked:
    if not groq_api_key:
        with left_col:
            st.warning("Configure GROQ_API_KEY in your local .env file or Streamlit Cloud secrets before analyzing.")
    elif not blood_report.strip():
        with left_col:
            st.warning("Please paste a blood work report before analyzing.")
    else:
        with st.spinner("Analyzing your blood work..."):

            # Stage 1: Extract and flag abnormal values
            extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""
            extraction_response = llm.invoke(extraction_prompt)
            extracted_values = extraction_response.text

            # Stage 2: Health summary and Indian diet plan
            diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, provide two clearly separated sections:

SECTION 1 - HEALTH SUMMARY:
Write 4-5 lines explaining the patient's condition in simple, non-technical language.

SECTION 2 - INDIAN DIET PLAN:
List foods to eat more of and foods to avoid, using commonly available Indian foods
like dal, sabzi, roti, rice, etc. Keep it practical and concise.

Blood Work Analysis:
{extracted_values}
"""
            diet_response = llm.invoke(diet_prompt)
            full_response = diet_response.text

        # Split response into two sections
        if "SECTION 2" in full_response:
            parts = full_response.split("SECTION 2")
            health_summary = parts[0].replace("SECTION 1 - HEALTH SUMMARY:", "").replace("SECTION 1", "").strip()
            diet_plan = ("SECTION 2" + parts[1]).replace("SECTION 2 - INDIAN DIET PLAN:", "").replace("SECTION 2", "").strip()
        else:
            health_summary = full_response
            diet_plan = ""

        # Render generated Markdown inside themed, scrollable output containers.
        with health_box:
            st.markdown(health_summary)
        with diet_box:
            st.markdown(diet_plan if diet_plan else full_response)
