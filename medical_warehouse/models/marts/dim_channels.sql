-- models/marts/dim_channels.sql
{{ config(materialized='table', schema='marts') }}

SELECT DISTINCT
    channel_name AS channel_id,
    channel_name AS channel_name,
    NULL::TEXT AS channel_description,  -- Placeholder, enrich later
    NULL::INTEGER AS follower_count     -- Placeholder, enrich later
FROM {{ ref('stg_telegram_messages') }}