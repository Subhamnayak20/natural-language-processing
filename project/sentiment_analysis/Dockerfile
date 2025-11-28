# Use a small, supported Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install git and basic build tools required by some pip packages
# clean apt lists to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better layer caching
# NOTE: your file is named 'requirement.txt' according to what you showed
COPY requirement.txt ./requirement.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the project into the image
COPY . .

# Expose the port your app runs on
EXPOSE 8080

# Start the Gradio app
CMD ["python", "app.py"]

