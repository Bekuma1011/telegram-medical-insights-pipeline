SELECT
    message_id,
    image_filename,
    COUNT(*) AS count
FROM {{ ref('fct_image_detections') }}
GROUP BY message_id, image_filename
HAVING COUNT(*) > 1