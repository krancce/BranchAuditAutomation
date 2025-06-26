from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import traceback
import logging

# Import the main function from your script
from audit_v2 import evaluate_store
from photoRetriever import get_failed_section_codes
import threading 

app = FastAPI(title="Branch Audit Automation API")

# Set up audit process log
LOG_FILE = 'logs/eval_api.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class FullAuditRequest(BaseModel):
    store_id: int
    month: int
    model: Optional[str] = "o4-mini"
    quarter: Optional[int] = 3

class FailedSectionsAuditRequest(BaseModel):
    store_id: int
    month: int
    quarter: Optional[int] = 3
    model: Optional[str] = "o4-mini"

def log_api_event(store_id, status):
    logging.info(f"Store {store_id}: {status}")

@app.post("/audit/store")
def run_full_audit(req: FullAuditRequest):
    try:
        log_api_event(req.store_id, "STARTED")
        year = datetime.now().year
        quarter = req.quarter if hasattr(req, "quarter") and req.quarter is not None else 3
        model = req.model or "o4-mini"
        start_date = f"{year}-{req.month:02d}-01"
        evaluate_store(
            store_id=req.store_id,
            quarter=quarter,
            model=model,
            start_date=start_date,
            month=req.month,
            year=year
        )
        log_api_event(req.store_id, "FINISHED")
        return {
            "status": "ok",
            "detail": f"Audit triggered for store {req.store_id} ({req.month:02d})."
        }
    except Exception as e:
        log_api_event(req.store_id, f"ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audit/failed_sections")
def run_failed_sections_audit(req: FailedSectionsAuditRequest):
    try:
        log_api_event(req.store_id, "FAILED SECTIONS AUDIT STARTED")
        year = datetime.now().year
        quarter = req.quarter or 3
        start_date = f"{year}-{req.month:02d}-01"
        model = req.model or "o4-mini"

        # Instead of querying the DB here, call your new function
        failed_section_codes = get_failed_section_codes(req.store_id, quarter)
        if not failed_section_codes:
            return {"status": "ok", "detail": "No failed sections found for this store in this quarter."}

        # Threaded evaluation for all failed sections
        results = []
        def evaluate_section(section_code):
            try:
                evaluate_store(
                    store_id=req.store_id,
                    quarter=quarter,
                    model=model,
                    start_date=start_date,
                    month=req.month,
                    year=year,
                    section_code=section_code
                )
                results.append({"section": section_code, "status": "success"})
            except Exception as e:
                results.append({"section": section_code, "status": f"error: {str(e)}"})

        threads = []
        for section in failed_section_codes:
            t = threading.Thread(target=evaluate_section, args=(section,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        log_api_event(req.store_id, f"FAILED SECTIONS AUDIT FINISHED: {failed_section_codes}")

        return {
            "status": "ok",
            "failed_sections_evaluated": failed_section_codes,
            "results": results
        }
    except Exception as e:
        log_api_event(req.store_id, f"FAILED SECTIONS AUDIT ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))