{{ config(materialized='view', schema='staging') }}

SELECT
    channel_name,
    message_id,
    image_filename,
    detected_object_class,
    confidence_score,
    detection_timestamp
FROM raw.image_detections
WHERE message_id IS NOT NULL
  AND detected_object_class IS NOT NULL
  AND confidence_score IS NOT NULL