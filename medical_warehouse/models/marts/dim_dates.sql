-- models/marts/dim_dates.sql
{{ config(materialized='table', schema='marts') }}

WITH date_range AS (
    SELECT generate_series(
        (SELECT MIN(message_date) FROM {{ ref('stg_telegram_messages') }}),
        (SELECT MAX(message_date) FROM {{ ref('stg_telegram_messages') }}),
        INTERVAL '1 day'
    ) AS post_date
)
SELECT
    ROW_NUMBER() OVER () AS date_id,
    post_date,
    TO_CHAR(post_date, 'Day') AS day_of_week,
    EXTRACT(WEEK FROM post_date) AS week_number,
    TO_CHAR(post_date, 'Month') AS month,
    EXTRACT(YEAR FROM post_date) AS year
FROM date_range