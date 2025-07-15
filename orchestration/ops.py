# D:\telegram-medical-insights-pipeline\orchestration\ops.py
import os
import subprocess
import logging
from dagster import op, Out
from dotenv import load_dotenv
from src.scrape_telegram import scrape_channel
from src.load_to_postgres import load_to_postgres
from src.process_images_yolo import process_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

@op(out=Out(bool))
def scrape_telegram_data(context):
    logger.info("Starting Telegram data scraping")
    try:
        channels = ['@CheMed123', '@lobelia4cosmetics', '@tikvahpharma']
        for channel in channels:
            scrape_channel(channel)
        logger.info("Telegram data scraping completed")
        return True
    except Exception as e:
        logger.error(f"Error in scrape_telegram_data: {e}")
        raise

@op(out=Out(bool))
def load_raw_to_postgres(context):
    logger.info("Starting data loading to PostgreSQL")
    try:
        load_to_postgres()
        logger.info("Data loading to PostgreSQL completed")
        return True
    except Exception as e:
        logger.error(f"Error in load_raw_to_postgres: {e}")
        raise

@op(out=Out(bool))
def run_yolo_enrichment(context):
    logger.info("Starting YOLO image processing")
    try:
        process_images()
        logger.info("YOLO image processing completed")
        return True
    except Exception as e:
        logger.error(f"Error in run_yolo_enrichment: {e}")
        raise

@op(out=Out(bool))
def run_dbt_transformations(context):
    logger.info("Starting dbt transformations")
    try:
        os.chdir('D:\\telegram-medical-insights-pipeline\\medical_warehouse')
        subprocess.run(['dbt', 'run', '--debug'], check=True)
        subprocess.run(['dbt', 'test'], check=True)
        logger.info("dbt transformations completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in run_dbt_transformations: {e}")
        raise
    finally:
        os.chdir('D:\\telegram-medical-insights-pipeline')