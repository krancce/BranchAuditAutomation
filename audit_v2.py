import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import base64
import io
import asyncio
from PIL import Image
import openai
import os
import time
import subprocess
import sys

from photoRetriever import PhotoRetriever

# Constants
TESTFILES_DIR = Path("TestFiles")
PROMPT_DIR = TESTFILES_DIR / "Branch_Audit_User_Prompts_All_Sections"
GOLDSTANDARD_DIR = TESTFILES_DIR / "GoldStandards"
SYSTEM_PROMPT_FILE = TESTFILES_DIR / "Branch_Audit_SYSTEM_PROMPT.txt"
OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs")

# Set up logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "audit_v2.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_warning(msg: str, error_log_path: str = "logs/error_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] ⚠️ {msg}"
    print(full_msg)
    os.makedirs("logs", exist_ok=True)
    with open(error_log_path, "a", encoding="utf-8") as errlog:
        errlog.write(full_msg + "\n")


# === Utility Functions ===

def resize_image_bytes(image_bytes: bytes, target_longer_side: int = 1000) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size

    if width >= height:
        w_new = target_longer_side
        h_new = int((height / width) * w_new)
    else:
        h_new = target_longer_side
        w_new = int((width / height) * h_new)

    img = img.resize((w_new, h_new))

    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG")
    return buf.getvalue()


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def load_system_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_user_prompt(section_code: str) -> str:
    prompt_path = PROMPT_DIR / f"{section_code}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

async def evaluate_section_async(client, task, model, start_date):
    messages = [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": [
            {"type": "text", "text": task["user_prompt"]},
            *[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in (task["gold_images"] + task["submission_images"])
            ]
        ]}
    ]

    for attempt in range(1, 4):  # Retry up to 3 times
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                seed=42,
                extra_headers={"x-openai-start-date": start_date}
            )

            content = response.choices[0].message.content


            # Attempt to extract as JSON object
            try:
                parsed_args = json.loads(content.strip().strip("```json").strip("```"))
                return {
                    "section": task["section"],
                    "result": json.dumps(parsed_args)  # still save as JSON string
                }
            except (json.JSONDecodeError, AttributeError, TypeError) as parse_err:
                log_warning(f"❌ Failed to parse tool arguments as JSON for store {task['store_id']}, section {task['section']}: {parse_err}")
                return None  # silently skip later


        except Exception as e:
            error_type = type(e).__name__
            http_status = getattr(e, 'http_status', None)

            if http_status:
                log_warning(
                    f"[Attempt {attempt}/3] ❌ {error_type} (HTTP {http_status}) for store {task['store_id']}, section {task['section']}: {e}"
                )
            else:
                log_warning(
                    f"[Attempt {attempt}/3] ❌ {error_type} for store {task['store_id']}, section {task['section']}: {e}"
                )

            if attempt == 3:
                log_warning(f"❌ Failed after 3 retries: Store {task['store_id']}, Section {task['section']}")
                return {
                    "section": task["section"],
                    "result": f"❌ API failure after 3 retries: {str(e)}"
                }

def submit_async_tasks(task_list, model: str, start_date: str):
    async def runner():
        client = openai.AsyncOpenAI()
        async with client:
            tasks = [evaluate_section_async(client, task, model, start_date) for task in task_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    logging.info("Starting async evaluation...")
    return asyncio.run(runner())

def track_cost(model_name: str, task_count: int):
    cost_log_file = LOG_DIR / "cost_log.json"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "tasks_submitted": task_count
    }

    if cost_log_file.exists():
        with open(cost_log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(cost_log_file, "w") as f:
        json.dump(logs, f, indent=2)

def generate_summary(store_id, results, submission_time, evaluation_time, total_time):
    passed_sections = []
    failed_sections = []
    for section, result in results.items():
        try:
            if isinstance(result, dict) and result.get("pass") is True:
                passed_sections.append(section)
            else:
                failed_sections.append(section)

        except Exception as e:
            log_warning(f"⚠️ Error reading result for section {section}: {e}")
            failed_sections.append(section)

    total = len(results)
    passed = len(passed_sections)
    failed = len(failed_sections)
    pass_rate = (passed / total * 100) if total else 0

    summary = (
        f"\n📋 Store Summary: {store_id:03d} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*60}\n"
        f"  Sections Evaluated: {total}\n"
        f"  ✅ Passed: {passed} — {passed_sections}\n"
        f"  ❌ Failed: {failed} — {failed_sections}\n"
        f"  Pass Rate: {pass_rate:.1f}%\n"
        f"{'+'*60}\n"
        f"  🕐 Time Breakdown:\n"
        f"     📸  Photo Retrieval: {submission_time:.2f} sec\n"
        f"     ⚙️  Evaluation Time: {evaluation_time:.2f} sec\n"
        f"     ⏱️  Total Elapsed:   {total_time:.2f} sec\n"

    )

    with open("logs/summary_log.txt", "a", encoding="utf-8") as logf:
        logf.write(summary + "\n")

# === Main Evaluation ===

def evaluate_store(store_id: int, quarter: int, model: str, start_date: str, month: int, year: int, section_code: str = None):
    logging.info(f"{'='*60}\n")
    if section_code:
        logging.info(f"Starting evaluation for Store {store_id:03d}, Q{quarter}, Model: {model}, Section: {section_code.upper()}")
    else:
        logging.info(f"Starting evaluation for Store {store_id:03d}, Q{quarter}, Model: {model}, All Sections")
    retriever = PhotoRetriever(store_id, quarter)
    photo_start = time.time()
    photos_by_section = retriever.retrieve_all_photos()
    photo_end = time.time()

    submission_time = photo_end - photo_start


    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    system_prompt = system_prompt.replace("required_start_date = YYYY-MM-DD", f"required_start_date = {start_date}")


    all_tasks = []

    # Filter to specific section if provided
    target_sections = {section_code.upper()} if section_code else photos_by_section.keys()

    for section_code, section_submissions in photos_by_section.items():
        if section_code not in target_sections:
            continue
        try:
            user_prompt_template = load_user_prompt(section_code)
        except FileNotFoundError:
            log_warning(f"Missing prompt for section {section_code}")
            continue

        gold_dir = GOLDSTANDARD_DIR / section_code
        gold_images = []
        for gold_file in gold_dir.iterdir():
            if gold_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                with open(gold_file, "rb") as img_f:
                    resized = resize_image_bytes(img_f.read())
                    gold_images.append(encode_image_bytes(resized))

        if not gold_images:
            log_warning(f"No gold standard images found for section {section_code}")
            continue

        # --- Loop over all pending submissions for this section ---
        for submission in section_submissions:
            submission_payloads = []
            for photo in submission["photos"]:
                if not photo.get("image_bytes"):
                    log_warning(f"Missing submission photo bytes for Store {store_id:03d}, Section {section_code}")
                    continue
                image_bytes = resize_image_bytes(photo["image_bytes"])
                submission_payloads.append(encode_image_bytes(image_bytes))

            # Dynamically insert the store justification comment:
            store_comment = submission.get("store_comment", "")
            user_prompt = user_prompt_template.replace(
                "store_justification_comments:",
                f"store_justification_comments:\n  {store_comment or 'None provided'}"
            )

            task = {
                "store_id": f"{store_id:03d}",
                "section": section_code,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "gold_images": gold_images,
                "submission_images": submission_payloads,
                # Optionally for logging/debugging:
                #"submission_id": submission.get("submission_id")
            }
            all_tasks.append(task)


    logging.info(f"Submitting {len(all_tasks)} tasks to OpenAI for evaluation...")
    results = submit_async_tasks(all_tasks, model=model, start_date=start_date)

    logging.info(f"Completed evaluations for Store {store_id:03d}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / model
    model_dir.mkdir(exist_ok=True)

    output_path = model_dir / f"store{store_id:03d}.json"
    formatted_results = {}
    for task in results:
        if task is None:
            continue  # Skip failed tasks

        section = task["section"]
        result = task["result"]

        if isinstance(result, str) and result.strip().startswith("{"):
            try:
                parsed_result = json.loads(result)
                # If nested with store ID like "001", flatten it
                if isinstance(parsed_result, dict):
                    # Try to flatten any nested store-level key
                    keys = list(parsed_result.keys())
                    if keys[0] in (f"{store_id:03d}", "000", "STORE_ID") or keys[0].isdigit():
                        parsed_result = parsed_result[keys[0]]

                formatted_results[section] = parsed_result

            except json.JSONDecodeError as e:
                log_warning(f"❌ JSON decode error for store {task['store_id']}, section {section}: {e}")
                log_warning(f"❓ Raw result for section {section}: {result}")
        elif result is None or (isinstance(result, str) and not result.strip()):
            log_warning(f"⚠️ Empty result for section {section}")
        else:
            log_warning(f"⚠️ Invalid non-JSON result for section {section}: {result}")


    # Load existing results if file already exists
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}


    # Update or append sections directly without nesting store_id
    existing_data.update(formatted_results)


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)
    
    # === Compute store score and pass to uploader ===

    try:
        # After writing, reload the whole output file so we have ALL results
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

        total_sections = 25
        passed_sections = sum(1 for sec in all_results.values() if sec.get("pass") is True)
        store_score = round((passed_sections / total_sections) * 100, 2) if total_sections else 0.0  

        subprocess.run([
            sys.executable, "upLoader.py",
            "--store_id", str(store_id),
            "--quarter", str(quarter),
            "--score", str(store_score),
            "--month", str(month),
            "--year", str(year)
        ], check=True)

    except Exception as e:
        log_warning(f"⚠️ Failed to compute or upload store score: {e}")
    
    logging.info(f"Results saved to {output_path}")
    # track_cost(model, len(all_tasks))
    # === Evaluation summary log ===
    evaluation_end = time.time()
    evaluation_time = evaluation_end - photo_end
    total_time = evaluation_end - photo_start
    generate_summary(store_id, all_results, submission_time, evaluation_time, total_time)


    # NEW: Save reference image paths per section to a log file
    ENABLE_REFERENCE_LOG = False # temp disabled for now...
    # ...
    if ENABLE_REFERENCE_LOG:
        reference_log_path = LOG_DIR / "reference_photo_log.txt"
        with open(reference_log_path, "a", encoding="utf-8") as ref_log:
            ref_log.write(f"\n=== Store {store_id:03d} Reference Image Log ===\n")
            for section_code, submission_photos in photos_by_section.items():
                ref_log.write(f"\n[Section {section_code}]\n")
                for submission in submission_photos:
                    ref_log.write(f"  - DB Path: {submission.get('original_path', 'N/A')} → Renamed: {submission['filename']}\n")
            ref_log.write("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_id", type=int, help="Store ID (optional)")
    parser.add_argument("--model", type=str, default="o4-mini")
    parser.add_argument("--start-date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--auto", action="store_true", help="Run evaluation for all qualifying stores automatically")
    parser.add_argument("--section", type=str, help="Evaluate a single section (e.g., B2, D4)")
    parser.add_argument("--month", type=int, required=True, help="Month of automated evaluation (1-12)")
    parser.add_argument("--year", type=int, required=True, help="Year of evaluation (e.g. 2025)")
    parser.add_argument("--quarter", type=int, default=3, help="Quarter number (1-4, default: 3)")

    args = parser.parse_args()

    if args.auto:
        print(f"📦 Fetching all stores with 25 section submissions for Q{args.quarter}...")
        store_ids = PhotoRetriever.get_stores_with_all_sections(args.quarter)
        print(f"🧾 Found {len(store_ids)} stores. Starting batch evaluation...\n")
        for store_id in store_ids:
            evaluate_store(
                store_id=store_id,
                quarter=args.quarter,
                model=args.model,
                start_date=args.start_date,
                month = args.month,
                year = args.year
            )

    elif args.store_id and args.section:
        # NEW: Single-section evaluation mode
        evaluate_store(
            store_id=args.store_id,
            quarter=args.quarter,
            model=args.model,
            start_date=args.start_date,
            section_code=args.section.upper(),
            month=args.month,
            year=args.year
        )
    elif args.store_id:
        # Existing: Full-store evaluation mode
        evaluate_store(
            store_id=args.store_id,
            quarter=args.quarter,
            model=args.model,
            start_date=args.start_date,
            month=args.month,
            year=args.year
        )
    else:
        print("❌ Please provide either --auto or --store_id [and optionally --section].")

