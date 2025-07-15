# Telegram Medical Insights Pipeline

Overview

The Telegram Medical Insights Pipeline is a data engineering project designed to collect, process, and analyze medical-related data from Telegram channels (@CheMed123, @lobelia4cosmetics, @tikvahpharma). The pipeline scrapes messages and images, loads them into a PostgreSQL database, enriches image data with YOLOv8 object detection, transforms data into a star schema using dbt, exposes analytical endpoints via a FastAPI application, and orchestrates the workflow with Dagster for robust scheduling and monitoring.

Prerequisites
Python: 3.9+
PostgreSQL: 13+ with database medical_warehouse (user: admin, password: securepassword, host: localhost, port: 5432)
Telegram API: API key and hash for scraping
YOLOv8: Pre-trained model for image processing
dbt: Installed with pip install dbt-postgres
Dagster: Installed with pip install dagster, dagster-webserver

1. Setup
Clone the Repository:
git clone <repository-url>
2. Set Up Virtual Environment:
python -m venv venv
.\venv\Scripts\activate
3. Install Dependencies:
pip install -r requirements.txt
4. Configure Environment Variables: Create a .env file in the project root:

- TELEGRAM_API_ID=<your-api-id>
- TELEGRAM_API_HASH=<your-api-hash>
- PHONE=<your phone number >
- DB_HOST=localhost
- DB_PORT=5432
- DB_NAME=<database name>
- DB_USER=<user>
- DB_PASSWORD=<password>

## pipeline Components
 ### Telegram Scraper (src/telegram_scraper.py):
Scrapes messages and images from @CheMed123, @lobelia4cosmetics, @tikvahpharma.
Stores data in data/raw/telegram_messages/<channel_name>.
### PostgreSQL Loader (src/load_to_postgres.py):
Loads raw data into raw.telegram_messages and raw.image_detections.
### YOLO Enrichment (src/process_images_yolo.py):
Processes images using YOLOv8 to detect medical objects.
Stores results in raw.image_detections.
### dbt Transformations (medical_warehouse/):
Transforms raw data into a star schema:
staging.stg_telegram_messages
staging.stg_image_detections
marts.dim_channels
marts.dim_dates
marts.fct_messages
marts.fct_image_detections
### FastAPI Application (api/):
Exposes endpoints:
- GET /api/reports/top-products?limit=10: Top products by mention count.
- GET /api/channels/{channel_name}/activity: Channel posting activity.
- GET /api/search/messages?query=<query>: Search messages by keyword.
### Dagster Orchestration (orchestration/):
Orchestrates the pipeline with ops for scraping, loading, YOLO processing, and dbt transformations.

### Run FastAPI:
uvicorn api.main:app --host 0.0.0.0 --port 8000
### To access Endpoints
http://localhost:8000/docs

