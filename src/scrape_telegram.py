import os
import json
import logging
import asyncio
from pathlib import Path
from telethon.sync import TelegramClient
from telethon.tl.types import InputMessagesFilterPhotos
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='logs/scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Telegram API credentials from .env
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")
phone = os.getenv("PHONE")

# Data lake directory
DATA_LAKE_PATH = Path("../data/raw/telegram_messages")

# List of Telegram channels to scrape
CHANNELS = [
    "@CheMed123",
    "@lobelia4cosmetics",
    "@tikvahpharma"
   ]

async def scrape_channel(client, channel, max_messages=300):
    """Scrape up to max_messages messages and images from a Telegram channel."""
    try:
        entity = await client.get_entity(channel)
        channel_name = entity.username or str(entity.id)
        logger.info(f"Scraping channel: {channel_name}")

        # Create directory for channel
        channel_path = DATA_LAKE_PATH / channel_name
        channel_path.mkdir(parents=True, exist_ok=True)

        # Scrape text messages
        message_count = 0
        async for message in client.iter_messages(
            entity,
            min_id=0,
            reverse=True,
            filter=None,
            wait_time=1,
            limit=max_messages
        ):
            date_str = message.date.strftime("%Y-%m-%d")
            file_path = channel_path / f"{date_str}.json"

            # Prepare message data
            message_data = {
                "message_id": message.id,
                "date": message.date.isoformat(),
                "text": message.text,
                "sender_id": message.sender_id,
                "media": bool(message.media),
                "views": message.views,
                "forwards": message.forwards
            }

            # Append to JSON file
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump(message_data, f, ensure_ascii=False)
                f.write('\n')

            logger.info(f"Saved message {message.id} from {channel_name} for {date_str}")
            message_count += 1
            if message_count >= max_messages:
                break

        # Scrape images (for specified channels)
        if channel in ["@CheMed123", "@lobelia4cosmetics"]:
            image_path = channel_path / "images"
            image_path.mkdir(parents=True, exist_ok=True)

            async for message in client.iter_messages(
                entity,
                min_id=0,
                reverse=True,
                filter=InputMessagesFilterPhotos,
                wait_time=1,
                limit=max_messages
            ):
                date_str = message.date.strftime("%Y-%m-%d")
                image_file = image_path / f"{date_str}_{message.id}.jpg"
                await message.download_media(file=str(image_file))
                logger.info(f"Saved image {message.id} from {channel_name} for {date_str}")

    except Exception as e:
        logger.error(f"Error scraping {channel}: {str(e)}")

async def main():
    """Main function to initialize client and scrape channels."""
    try:
        async with TelegramClient('session_name', api_id, api_hash) as client:
            await client.start(phone=phone)
            logger.info("Telegram client started")

            # Scrape each channel
            for channel in CHANNELS:
                await scrape_channel(client, channel, max_messages=300)

    except Exception as e:
        logger.error(f"Main loop error: {str(e)}")

if __name__ == "__main__":
    # Ensure data lake directory exists
    DATA_LAKE_PATH.mkdir(parents=True, exist_ok=True)
    asyncio.run(main())