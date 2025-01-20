from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import datetime
import os

def run_scraper():
    print(f"Starting job scraper at {datetime.datetime.now()}")
    try:
        # Run the scraper
        subprocess.run(['python', 'fin.py'], check=True)
        # Run the analysis
        subprocess.run(['python', 'analysis.py'], check=True)
        print("Scraping and analysis completed successfully")
    except Exception as e:
        print(f"Error running scraper: {str(e)}")

# Create scheduler
scheduler = BlockingScheduler()
# Schedule job to run every Monday at 1 AM
scheduler.add_job(run_scraper, 'cron', day_of_week='mon', hour=1)

if __name__ == "__main__":
    print("Starting scheduler...")
    # Run once immediately when starting
    run_scraper()
    # Start the scheduler
    scheduler.start() 