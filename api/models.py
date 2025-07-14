from dataclasses import dataclass
from datetime import datetime, date

@dataclass
class ProductReport:
    channel_name: str
    message_text: str
    mention_count: int

@dataclass
class ChannelActivity:
    post_date: date
    week_number: int
    post_count: int

@dataclass
class MessageSearch:
    channel_name: str
    message_text: str
    message_timestamp: datetime
    detected_object_class: str | None
    confidence_score: float | None