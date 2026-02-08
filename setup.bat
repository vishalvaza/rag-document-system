@echo off
echo ================================================
echo RAG Document QA - Setup Script
echo ================================================
echo.

echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ================================================
echo Setup complete!
echo ================================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env and configure your LLM:
echo    - For OpenAI: Add OPENAI_API_KEY
echo    - For Ollama: Set OLLAMA_BASE_URL and OLLAMA_MODEL
echo 2. Add your documents to the data/ folder
echo 3. Run: python ingest.py
echo 4. Run: streamlit run app.py
echo.
echo To activate the environment in the future, run:
echo   .venv\Scripts\activate
echo.
pause
