from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import traceback
import logging

# Import the main function from your script
from audit_v2 import evaluate_store

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
    quarter: int
    model: Optional[str] = "o4-mini"
    start_date: Optional[str] = "2025-05-01"

class SectionAuditRequest(FullAuditRequest):
    section_code: str

def log_api_event(store_id, status):
    logging.info(f"Store {store_id}: {status}")

@app.post("/audit/store")
def run_full_audit(req: FullAuditRequest):
    try:
        log_api_event(req.store_id, "STARTED")
        start_date = req.start_date or datetime.now().strftime("%Y-%m-%d")
        evaluate_store(
            store_id=req.store_id,
            quarter=req.quarter,
            model=req.model,
            start_date=start_date
        )
        log_api_event(req.store_id, "FINISHED")
        return {
            "status": "ok",
            "detail": f"Audit triggered for store {req.store_id} (Q{req.quarter})."
        }
    except Exception as e:
        log_api_event(req.store_id, f"ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audit/section")
def run_section_audit(req: SectionAuditRequest):
    try:
        log_api_event(req.store_id, f"SECTION {req.section_code.upper()} STARTED")
        start_date = req.start_date or datetime.now().strftime("%Y-%m-%d")
        evaluate_store(
            store_id=req.store_id,
            quarter=req.quarter,
            model=req.model,
            start_date=start_date,
            section_code=req.section_code.upper()
        )
        log_api_event(req.store_id, f"SECTION {req.section_code.upper()} FINISHED")
        return {
            "status": "ok",
            "detail": f"Section audit triggered for store {req.store_id} (Q{req.quarter}), section {req.section_code.upper()}."
        }
    except Exception as e:
        log_api_event(req.store_id, f"SECTION {req.section_code.upper()} ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
