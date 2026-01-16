FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE=mangoAPI.settings

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libhdf5-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy the rest of the application
COPY . .

# Collect static files (skip check to avoid build failures)
RUN python manage.py collectstatic --noinput || true

# Create a simple startup script with better error handling
RUN echo '#!/bin/bash\nset -e\necho "Running migrations..."\npython manage.py migrate --noinput || echo "Migration warning - continuing"\necho "Creating superuser if needed..."\npython manage.py create_superuser || echo "Superuser exists or creation skipped"\nPORT=${PORT:-8000}\necho "Starting gunicorn on 0.0.0.0:$PORT"\nexec gunicorn mangoAPI.wsgi:application \\\n  --bind 0.0.0.0:$PORT \\\n  --workers 2 \\\n  --threads 4 \\\n  --worker-class gthread \\\n  --timeout 120 \\\n  --graceful-timeout 30 \\\n  --keep-alive 5 \\\n  --log-level info \\\n  --access-logfile - \\\n  --error-logfile -' > /app/start.sh
RUN chmod +x /app/start.sh

# Expose default port
EXPOSE 8000

# Use the startup script
CMD ["/app/start.sh"]