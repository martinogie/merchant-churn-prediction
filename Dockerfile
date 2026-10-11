FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY churn/ churn/
COPY examples/ examples/
# Train at build time so the image ships with a model.
RUN python -m churn.run
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "2", "churn.api:create_app()"]
