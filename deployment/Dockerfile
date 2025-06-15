# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Create a script to run the app
RUN echo '#!/bin/bash\nstreamlit run app/main.py --server.port=8501 --server.address=0.0.0.0' > /app/run.sh
RUN chmod +x /app/run.sh

# Command to run the app
CMD ["/app/run.sh"]
