
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.db.session import SessionLocal
from src.db.repositories.ticker_repo import TickerRepository
from src.db.repositories.snapshot_repo import SnapshotRepository
from src.db.models.ticker import Ticker
from src.providers.yahoo.client import YFinanceProvider
from src.providers.fx_rates import FX_PROVIDER
from src.ingestion.service import IngestionService
from src.ingestion.config import IngestionConfig
from src.ingestion.report import IngestionReport

async def populate(limit: int = None):
    print("Connecting to database...")
    db = SessionLocal()
    
    # Get all tickers
    if limit:
        tickers_query = db.query(Ticker.ticker).limit(limit).all()
    else:
        tickers_query = db.query(Ticker.ticker).all()
    
    tickers = [t[0] for t in tickers_query]
    
    db.close()
    
    total_tickers = len(tickers)
    print(f"Found {total_tickers} tickers in the database.")
    
    if total_tickers == 0:
        print("No tickers found! Did you run scripts/ingest_tickers.py?")
        return

    # In output, let's limit for testing if user wants, but default to all
    # For this run, we'll do ALL as requested "populate the snapshot db"
    
    print("Initializing services...")
    # Re-open session for repos (though service might not need persistent session if it handles it per op, 
    # but Repos take a session in init. Ideally we should manage scope.)
    # The repositories take a session. IngestionService takes repos. 
    # The current repo implementation keeps one session open. This might be bad for 7000 writes?
    # SQLAlchemy session handles transaction. 
    # We should probably commit periodically or `upsert` already commits.
    # Looking at `TickerRepo.upsert`, it does `self.session.commit()`. So it commits per item.
    # That is safe but slower.
    
    db = SessionLocal()
    ticker_repo = TickerRepository(db)
    snapshot_repo = SnapshotRepository(db)
    provider = YFinanceProvider()
    
    service = IngestionService(
        provider=provider,
        fx_provider=FX_PROVIDER,
        snapshot_repo=snapshot_repo,
        ticker_repo=ticker_repo,
        config=IngestionConfig()
    )
    
    print(f"Starting ingestion for {len(tickers)} tickers in batches...")
    
    # Process in batches of 100
    batch_size = 100
    report = IngestionReport() # We need to aggregate reports or just let service append to one if we passed it?
    # Service methods create new report. Let's accumulate.
    # A cleaner way is to make service accept an existing report or just sum them up.
    # IngestionReport has simple counters and a dict. We can merge.
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        print(f"Processing batch {i // batch_size + 1} ({len(batch)} tickers)...")
        
        batch_report = await service.ingest_tickers(batch)
        
        # Merge report
        report.attempted.extend(batch_report.attempted)
        report.successes.update(batch_report.successes)
        report.failures.update(batch_report.failures)
        
        # Optional: Sleep to be nice to API? Rate limiter handles it, but maybe safer.
        # await asyncio.sleep(1)
    
    print("\nIngestion Report:")
    print(f"Successes: {report.success_count}")
    print(f"Failures: {report.failure_count}")
    
    if report.failures:
        print("Failures (first 5):", list(report.failures.items())[:5])

    db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of tickers to process")
    args = parser.parse_args()
    
    asyncio.run(populate(limit=args.limit))
