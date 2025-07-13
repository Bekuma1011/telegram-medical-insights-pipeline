-- tests/no_duplicate_messages.sql
SELECT
    channel_id,
    message_id,
    post_date_id,
    COUNT(*) AS count
FROM {{ ref('fct_messages') }}
GROUP BY channel_id, message_id, post_date_id
HAVING COUNT(*) > 1