{{ config(materialized='table', schema='marts') }}

SELECT
    ROW_NUMBER() OVER () AS detection_key,
    s.message_id,
    s.channel_name AS channel_id,
    m.message_key AS message_key,
    s.image_filename,
    s.detected_object_class,
    s.confidence_score,
    s.detection_timestamp
FROM {{ ref('stg_image_detections') }} s
JOIN {{ ref('fct_messages') }} m
    ON s.message_id = m.message_id
    AND s.channel_name = m.channel_id
JOIN {{ ref('dim_channels') }} c
    ON s.channel_name = c.channel_id