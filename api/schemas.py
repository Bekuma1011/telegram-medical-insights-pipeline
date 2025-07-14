from pydantic import BaseModel
from datetime import datetime, date
from typing import Optional

class ProductReport(BaseModel):
    channel_name: str
    message_text: str
    mention_count: int

    class Config:
        from_attributes = True

class ChannelActivity(BaseModel):
    post_date: date
    week_number: int
    post_count: int

    class Config:
        from_attributes = True

class MessageSearch(BaseModel):
    channel_name: str
    message_text: str
    message_timestamp: datetime
    detected_object_class: Optional[str] = None
    confidence_score: Optional[float] = None

    class Config:
        from_attributes = True