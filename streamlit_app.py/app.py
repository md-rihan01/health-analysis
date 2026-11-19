from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html import escape
import os
import re
import time

from dotenv import load_dotenv
import streamlit as st
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from langchain_groq import ChatGroq
from streamlit.errors import StreamlitSecretNotFoundError


load_dotenv()
st.set_page_config(page_title="Blood Work Analyzer", layout="wide")


def get_groq_api_key():
    """Read credentials locally from .env or from Streamlit Cloud secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets.get("GROQ_API_KEY")
    except StreamlitSecretNotFoundError:
        return None


def _duration_seconds(value):
    """Parse common Groq reset headers such as '2s' or '1m30s'."""
    if not value:
        return None
    value = str(value).strip().lower()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", value)
    if not parts:
        return None
    multipliers = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(amount) * multipliers[unit] for amount, unit in parts)


def _retry_delay(error, attempt):
    """Use Groq's retry headers when present, otherwise use short backoff."""
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    headers = {str(key).lower(): value for key, value in headers.items()}

    delay = _duration_seconds(headers.get("retry-after-ms"))
    if delay is None:
        delay = _duration_seconds(headers.get("retry-after"))
    if delay is None and headers.get("retry-after"):
        try:
            retry_at = parsedate_to_datetime(headers["retry-after"])
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            pass
    if delay is None:
        delay = _duration_seconds(headers.get("x-ratelimit-reset-requests"))
    if delay is None:
        delay = _duration_seconds(headers.get("x-ratelimit-reset-tokens"))
    if delay is None:
        delay = min(2**attempt, 4)
    return delay


class RateLimitCooldown(Exception):
    """A rate limit whose suggested wait is too long for an automatic retry."""

    def __init__(self, seconds):
        self.seconds = seconds


def invoke_with_rate_limit_retry(llm, prompt):
    """Retry a few times, but never sleep through a long quota reset."""
    max_retries = 2
    max_automatic_wait = 10

    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(prompt)
        except RateLimitError as error:
            if attempt >= max_retries:
                raise
            delay = _retry_delay(error, attempt)
            # If the service says the quota will take longer to reset, return
            # control to the user instead of holding a mobile session open.
            if delay > max_automatic_wait:
                raise RateLimitCooldown(delay) from error
            time.sleep(delay)


def mark_analysis_started():
    st.session_state.analysis_running = True


groq_api_key = get_groq_api_key()
llm = (
    ChatGroq(
        model="qwen/qwen3.8-27b",
        reasoning_format="parsed",
        api_key=groq_api_key,
        max_retries=0,
        timeout=30,
    )
    if groq_api_key
    else None
)

st.session_state.setdefault("analysis_running", False)

st.markdown("""
<style>
.scroll-box {
    height: 230px;
    overflow-y: auto;
    padding: 12px 16px;
    border: 1px solid #333;
    border-radius: 8px;
    background-color: #1e1e1e;
    font-size: 0.9rem;
    line-height: 1.6;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.scroll-box p, .scroll-box li {
    color: #e0e0e0;
}
.section-label {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 6px;
    color: #ffffff;
}
@media (max-width: 640px) {
    .scroll-box { height: auto; max-height: 320px; }
    button[kind="primary"] { min-height: 48px; }
    textarea { font-size: 16px !important; }
}
</style>
""", unsafe_allow_html=True)

st.title("Blood Work Analyzer")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Blood Work Report")
    with st.form("blood_work_form", clear_on_submit=False):
        blood_report = st.text_area(
            label="Paste your report below",
            height=500,
            placeholder="Paste your blood work report here...",
            label_visibility="collapsed",
        )
        analyze_clicked = st.form_submit_button(
            "Analyze",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running,
            on_click=mark_analysis_started,
        )

report_digest = sha256(blood_report.strip().encode("utf-8")).hexdigest()
cached_result = st.session_state.get("analysis_result")
if cached_result and cached_result["report_digest"] == report_digest:
    health_summary, diet_plan = cached_result["health_summary"], cached_result["diet_plan"]
else:
    health_summary, diet_plan = "", ""

with right_col:
    st.subheader("Health Summary")
    health_box = st.empty()
    health_box.markdown(
        f'<div class="scroll-box">{escape(health_summary).replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Suggested Diet Plan")
    diet_box = st.empty()
    diet_box.markdown(
        f'<div class="scroll-box">{escape(diet_plan).replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

if analyze_clicked:
    try:
        if not groq_api_key:
            with left_col:
                st.warning(
                    "Configure GROQ_API_KEY in your local .env file or Streamlit Cloud secrets before analyzing."
                )
        elif not blood_report.strip():
            with left_col:
                st.warning("Please paste a blood work report before analyzing.")
        elif cached_result and cached_result["report_digest"] == report_digest:
            # Repeated mobile taps with the same report reuse this session's
            # result and do not make another paid API request.
            pass
        else:
            extraction_prompt = f"""
You are a careful medical data extraction assistant and clinical nutritionist.
Use only the blood report below. Do not infer, alter, or invent test names,
values, units, or reference ranges. For each reported test, preserve its
original value and range and label it HIGH, LOW, or NORMAL only when the
provided range supports that label. If no range is given, say "range not provided".

Return exactly these two sections:
SECTION 1 - HEALTH SUMMARY:
- First list reported tests as: Test: original value and unit | Status: HIGH/LOW/NORMAL/NOT CLASSIFIED | Reference: original range or "not provided".
- Clearly separate abnormal results from normal results.
- Then summarize the findings in 3-4 simple lines. Do not diagnose.

SECTION 2 - INDIAN DIET PLAN:
- Give concise, practical foods to eat more of and foods to limit, using common
  Indian foods where appropriate. Do not present the suggestions as treatment.

Blood report:
{blood_report}
"""
            with st.spinner("Analyzing your blood work..."):
                response = invoke_with_rate_limit_retry(llm, extraction_prompt)
                full_response = response.text
                if not isinstance(full_response, str) or not full_response.strip():
                    raise ValueError("The analysis service returned an empty response.")

            if "SECTION 2" not in full_response:
                raise ValueError("The analysis service returned an incomplete response.")

            parts = full_response.split("SECTION 2", 1)
            health_summary = parts[0].replace("SECTION 1 - HEALTH SUMMARY:", "").replace("SECTION 1", "").strip()
            diet_plan = parts[1].replace("- INDIAN DIET PLAN:", "").replace("INDIAN DIET PLAN:", "").strip()
            st.session_state.analysis_result = {
                "report_digest": report_digest,
                "health_summary": health_summary,
                "diet_plan": diet_plan,
            }
            health_box.markdown(
                f'<div class="scroll-box">{escape(health_summary).replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            diet_box.markdown(
                f'<div class="scroll-box">{escape(diet_plan).replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
    except RateLimitCooldown as error:
        with left_col:
            wait = max(1, round(error.seconds))
            st.warning(
                f"Groq is temporarily rate-limiting requests. Please wait about {wait} seconds and try again."
            )
    except RateLimitError:
        with left_col:
            st.warning(
                "Groq is currently rate-limiting requests or the API quota is exhausted. "
                "Please wait before trying again or check your Groq plan."
            )
    except AuthenticationError:
        with left_col:
            st.error("Groq could not authenticate the API key. Check GROQ_API_KEY in your .env file or Streamlit secrets.")
    except (APITimeoutError, APIConnectionError):
        with left_col:
            st.error("Groq could not be reached right now. Please check your connection and try again.")
    except APIStatusError:
        with left_col:
            st.error("Groq could not complete the analysis. Please try again later.")
    except ValueError:
        with left_col:
            st.error("The analysis response was incomplete. No results were shown; please try again.")
    except Exception:
        # Do not show exception details: they may contain request or patient data.
        with left_col:
            st.error("An unexpected error prevented the analysis. Please try again later.")
    finally:
        st.session_state.analysis_running = False

st.caption(
    "This AI-generated summary is for informational purposes only, not a diagnosis or treatment plan. "
    "Review results with a qualified healthcare professional."
)
