SELECT *
FROM {{ ref('fct_messages') }}
WHERE message_text = ''