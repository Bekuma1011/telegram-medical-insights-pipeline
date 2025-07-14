import os
import psycopg2
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv
import logging

# Setup logging
# Setup logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('../logs/process_images_yolo.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# PostgreSQL connection parameters
conn_params = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Data lake path for images
DATA_LAKE_PATH = Path("../data/raw/telegram_messages")

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

def create_image_detections_table(conn):
    """Create raw.image_detections table if it doesn't exist."""
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE SCHEMA IF NOT EXISTS raw;
            CREATE TABLE IF NOT EXISTS raw.image_detections (
                channel_name VARCHAR(255),
                message_id BIGINT,
                image_filename VARCHAR(255),
                detected_object_class VARCHAR(100),
                confidence_score FLOAT,
                detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        logger.info("Created raw.image_detections table")

def process_images():
    """Scan images in data lake, run YOLOv8, and store detections in PostgreSQL."""
    try:
        conn = psycopg2.connect(**conn_params)
        create_image_detections_table(conn)
        cursor = conn.cursor()
        
        # Scan data lake for images
        for channel_dir in DATA_LAKE_PATH.iterdir():
            if not channel_dir.is_dir():
                continue
            channel_name = channel_dir.name
            image_dir = channel_dir / "images"
            if not image_dir.exists():
                logger.info(f"No images directory for {channel_name}")
                continue
        
            for image_path in image_dir.glob("*.jpg"):  
                try:
                    # Extract message_id from filename (e.g., message_12345.jpg)
                    filename = image_path.name
                    message_id = int(filename.split('_')[1].split('.')[0])
                    # Run YOLOv8 inference
                    results = model(image_path)
                    detections = []
                    
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            class_name = model.names[class_id]
                            confidence = float(box.conf)
                            detections.append((channel_name, message_id, filename, class_name, confidence))
                                                 

                    # Insert detections into PostgreSQL
                    if detections:
                        query = """
                            INSERT INTO raw.image_detections (
                                channel_name, message_id, image_filename,
                                detected_object_class, confidence_score
                            ) VALUES (%s, %s, %s, %s, %s)
                        """
                        cursor.executemany(query, detections)
                        conn.commit()
                        logger.info(f"Processed {image_path}: {len(detections)} detections")
                    else:
                        logger.info(f"No detections for {image_path}")
                
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    conn.rollback()
             

        cursor.close()
        conn.close()

    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        raise

if __name__ == "__main__":
    process_images()