# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (important for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]