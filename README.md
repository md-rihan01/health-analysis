# Blood Work Analyzer

The Streamlit app is located at `streamlit_app.py/app.py`. It uses Groq to extract laboratory values from a pasted blood report and generate a health summary and practical Indian diet suggestions. The notebook workflows and sample report remain in this folder.

## Local Windows setup

Run these commands from the repository root in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r 2_health_analysis\streamlit_app.py\requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` and set your real key:

```dotenv
GROQ_API_KEY=your_actual_groq_api_key
```

Start the app from the repository root:

```powershell
python -m streamlit run 2_health_analysis\streamlit_app.py\app.py
```

Streamlit opens at `http://localhost:8501`. You can also put the key in `.streamlit/secrets.toml` using the format below; keep both secret files out of Git.

## Streamlit Community Cloud

Push the repository to GitHub, then create a Streamlit Community Cloud app using:

- **Repository:** your GitHub repository
- **Branch:** the branch containing this project
- **Main file path:** `2_health_analysis/streamlit_app.py/app.py`
- **Python version:** 3.14 (matching `.python-version`)

The app's adjacent `requirements.txt` supplies its dependencies. In the app's **Advanced settings → Secrets**, paste:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

Do not commit `.env` or `.streamlit/secrets.toml`. The checked-in `.env.example` and `.streamlit/secrets.toml.example` contain placeholders only.

## Notes

The app does not make an API call unless **Analyze** is clicked with a report and key configured. It retries explicit Groq rate-limit responses up to three total attempts with short exponential delays. Authentication, network, service, and unexpected failures are shown as user-friendly messages; no fallback or fabricated medical result is displayed.

This is an educational demonstration, not a diagnostic or treatment tool. AI results can be incomplete or wrong; discuss laboratory results and any diet or treatment changes with a qualified healthcare professional.
