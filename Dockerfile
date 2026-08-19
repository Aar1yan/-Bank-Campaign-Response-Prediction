FROM python:3.9-slim

WORKDIR /app

# Production-only deps - the notebook/dev tools in requirements.txt (jupyter,
# matplotlib, xgboost, shap, pytest, ...) aren't imported by the deployed app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Only the code and artifacts the running application actually needs
COPY src/ src/
COPY app/ app/
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
