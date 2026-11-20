from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from streamlit.errors import StreamlitSecretNotFoundError
from groq import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
import os
import time

load_dotenv()
st.set_page_config(page_title="Blood Work Analyzer", layout="wide")


def get_groq_api_key():
    """Read the key from Streamlit secrets first, then local environment."""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        secret_key = None

    return secret_key or os.getenv("GROQ_API_KEY")


def invoke_with_rate_limit_retry(llm, prompt, max_attempts=3):
    """Retry only explicit rate-limit responses, which were not accepted for processing."""
    for attempt in range(max_attempts):
        try:
            return llm.invoke(prompt)
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)


groq_api_key = get_groq_api_key()

st.markdown("""
<style>
.st-key-health-output,
.st-key-diet-output {
    max-width: 100%;
    border-color: var(--border-color);
    border-radius: 8px;
    color: var(--text-color);
    font-size: 0.9rem;
    line-height: 1.6;
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
.st-key-health-output h1,
.st-key-health-output h2,
.st-key-health-output h3,
.st-key-health-output strong,
.st-key-diet-output p,
.st-key-diet-output li,
.st-key-diet-output h1,
.st-key-diet-output h2,
.st-key-diet-output h3,
.st-key-diet-output strong {
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
    font-weight: 700;
}
@media (max-width: 640px) {
    .st-key-health-output,
    .st-key-diet-output {
        font-size: 0.88rem;
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
        st.error("GROQ_API_KEY is not configured. Add it to your local .env file or Streamlit Cloud app secrets.")
    elif not blood_report.strip():
        with left_col:
            st.warning("Please paste a blood work report before analyzing.")
    else:
        with st.spinner("Analyzing your blood work..."):
            try:
                llm = ChatGroq(
                    model="qwen/qwen3.8-27b",
                    reasoning_format="parsed",
                    api_key=groq_api_key,
                )

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
                extraction_response = invoke_with_rate_limit_retry(llm, extraction_prompt)
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
                diet_response = invoke_with_rate_limit_retry(llm, diet_prompt)
                full_response = diet_response.text

                # Split response into two sections
                if "SECTION 2" in full_response:
                    parts = full_response.split("SECTION 2")
                    health_summary = parts[0].replace("SECTION 1 - HEALTH SUMMARY:", "").replace("SECTION 1", "").strip()
                    diet_plan = ("SECTION 2" + parts[1]).replace("SECTION 2 - INDIAN DIET PLAN:", "").replace("SECTION 2", "").strip()
                else:
                    health_summary = full_response
                    diet_plan = ""

                # Keep model output as Markdown; Streamlit renders it without unsafe HTML.
                health_box.markdown(health_summary)
                diet_box.markdown(diet_plan if diet_plan else full_response)
            except RateLimitError:
                st.warning("Groq is rate limiting requests right now. Please wait a little and try again.")
            except AuthenticationError:
                st.error("Groq rejected the configured API key. Check GROQ_API_KEY in your local or cloud secrets.")
            except APIConnectionError:
                st.error("Could not connect to Groq. Check your internet connection and try again.")
            except APIStatusError:
                st.error("Groq could not complete the analysis. Please try again later.")
            except Exception:
                st.error("The analysis could not be completed. Please check the app configuration and try again.")
