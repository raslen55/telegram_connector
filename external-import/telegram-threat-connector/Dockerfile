FROM python:3.11-slim

# Copy the connector
COPY . /opt/opencti-connector-telegram
WORKDIR /opt/opencti-connector-telegram

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Entrypoint
ENTRYPOINT ["python3", "src/main.py"]
