# D:\telegram-medical-insights-pipeline\api\crud.py
from .database import get_db_connection
from .models import ProductReport, ChannelActivity, MessageSearch

def get_top_products(limit: int = 10):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    c.channel_name,
                    m.message_text,
                    COUNT(*) as mention_count
                FROM raw_marts.fct_messages m
                JOIN raw_marts.dim_channels c ON m.channel_id = c.channel_id
                GROUP BY c.channel_name, m.message_text
                ORDER BY mention_count DESC
                LIMIT %s
            """, (limit,))
            return [ProductReport(**row) for row in cursor.fetchall()]
    finally:
        conn.close()

def get_channel_activity(channel_name: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    d.post_date,
                    d.week_number,
                    COUNT(*) as post_count
                FROM raw_marts.fct_messages m
                JOIN raw_marts.dim_dates d ON m.post_date_id = d.date_id
                JOIN raw_marts.dim_channels c ON m.channel_id = c.channel_id
                WHERE c.channel_name = %s
                GROUP BY d.post_date, d.week_number
                ORDER BY d.post_date
            """, (channel_name,))
            return [ChannelActivity(**row) for row in cursor.fetchall()]
    finally:
        conn.close()

def search_messages(query: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    c.channel_name,
                    m.message_text,
                    m.message_timestamp,
                    d.detected_object_class,
                    d.confidence_score
                FROM raw_marts.fct_messages m
                JOIN raw_marts.dim_channels c ON m.channel_id = c.channel_id
                LEFT JOIN raw_marts.fct_image_detections d ON m.message_key = d.message_key
                WHERE m.message_text ILIKE %s
                ORDER BY m.message_timestamp DESC
            """, (f'%{query}%',))
            return [MessageSearch(**row) for row in cursor.fetchall()]
    finally:
        conn.close()