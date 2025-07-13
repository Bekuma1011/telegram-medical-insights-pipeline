-- models/marts/fct_messages.sql
{{ config(materialized='table', schema='marts') }}

SELECT
    ROW_NUMBER() OVER () AS message_key,
    s.channel_name AS channel_id,
    d.date_id AS post_date_id,
    s.message_id,
    s.message_timestamp,
    s.message_text,
    LENGTH(s.message_text) AS message_length,
    s.has_media,
    s.view_count,
    s.forward_count
FROM {{ ref('stg_telegram_messages') }} s
JOIN {{ ref('dim_dates') }} d
    ON s.message_date = d.post_date
JOIN {{ ref('dim_channels') }} c
    ON s.channel_name = c.channel_id