# 1. Use Python 3.10 as you requested
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only requirements first (for better caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy everything else (app.py, model folder, etc.)
COPY . .

# 6. Expose the port Hugging Face requires
EXPOSE 7860

# 7. Run the app using Gunicorn (better for APIs)
# Change 'app:app' if your Flask variable name isn't 'app'
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
