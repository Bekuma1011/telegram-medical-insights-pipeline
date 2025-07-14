from fastapi import FastAPI, HTTPException
from .crud import get_top_products, get_channel_activity, search_messages
from .schemas import ProductReport, ChannelActivity, MessageSearch
from typing import List

app = FastAPI(title="Kara Solutions Telegram Analytics API")

@app.get("/api/reports/top-products", response_model=List[ProductReport])
async def top_products(limit: int = 10):
    try:
        return get_top_products(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top products: {e}")

@app.get("/api/channels/{channel_name}/activity", response_model=List[ChannelActivity])
async def channel_activity(channel_name: str):
    try:
        activities = get_channel_activity(channel_name)
        if not activities:
            raise HTTPException(status_code=404, detail=f"No activity found for channel {channel_name}")
        return activities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching channel activity: {e}")

@app.get("/api/search/messages", response_model=List[MessageSearch])
async def search_messages(query: str):
    try:
        results = search_messages(query)
        if not results:
            raise HTTPException(status_code=404, detail=f"No messages found for query: {query}")
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching messages: {e}")