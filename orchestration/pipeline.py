from dagster import job, ScheduleDefinition, DefaultScheduleStatus
from .ops import scrape_telegram_data, load_raw_to_postgres, run_yolo_enrichment, run_dbt_transformations

@job
def telegram_pipeline():
    scrape_success = scrape_telegram_data()
    load_success = load_raw_to_postgres(scrape_success)
    yolo_success = run_yolo_enrichment(load_success)
    run_dbt_transformations(yolo_success)

# Schedule to run daily at 1 AM
daily_schedule = ScheduleDefinition(
    job=telegram_pipeline,
    cron_schedule="0 1 * * *",  # 1 AM daily
    default_status=DefaultScheduleStatus.STOPPED
)