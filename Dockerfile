FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Now copy remaining files
COPY . .

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "drowsiness:app", "--workers", "1", "--threads", "1"]
