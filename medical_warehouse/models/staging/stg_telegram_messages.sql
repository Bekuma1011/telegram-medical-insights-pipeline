{{ config(materialized='view', schema='staging') }}

SELECT
    channel_name,
    message_id,
    CAST(date AS TIMESTAMP) AS message_timestamp,
    CAST(message_date AS DATE) AS message_date,
    COALESCE(text, '') AS message_text,
    COALESCE(sender_id, 0) AS sender_id,
    COALESCE(media, FALSE) AS has_media,
    COALESCE(views, 0) AS view_count,
    COALESCE(forwards, 0) AS forward_count
FROM raw.telegram_messages
WHERE message_id IS NOT NULL
  AND (text IS NOT NULL AND text != '')