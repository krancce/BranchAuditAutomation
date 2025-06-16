import json
import argparse
import os
from PIL import Image
import openai
from pathlib import Path
from typing import List, Dict, Any
import io
import base64
import logging
from datetime import datetime
import time
from dotenv import load_dotenv
import httpx
import asyncio



load_dotenv(override=True) # pulls variables from .env into os.environ
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"✅ Python sees OPENAI_API_KEY = {os.getenv('OPENAI_API_KEY')[:10]}...")


# --- Configuration & Constants ---
MODEL_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "o4-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

MODEL_TOKEN_LIMIT = {
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    "o4-mini": 128000,  # Assume same limit as gpt-4o until confirmed otherwise
}


# Note: Check official OpenAI pricing for the most up-to-date values.
# Pricing for gpt-3.5-turbo can vary; I've used a common recent one.
# gpt-4o-mini pricing is stated as unofficial in the spec.

MAX_WIDTH = 800  # pixels
LOGS_DIR = Path("logs")
OUTPUTS_DIR = Path("outputs")
COST_LOG_FILE = LOGS_DIR / "cost_log.txt"
ERROR_LOG_FILE = LOGS_DIR / "error_log.txt"

# --- Logging Setup ---
def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=ERROR_LOG_FILE,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# --- Helper Functions ---
def ensure_dir_exists(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# --- Image Utilities ---
def resize_image(image_path: Path, max_width: int = MAX_WIDTH) -> bytes:
    """Resizes an image to a max width, maintaining aspect ratio, and returns JPEG bytes."""
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if not already (e.g., for PNGs with alpha or other modes)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        if img.width > max_width:
            ratio = max_width / float(img.width)
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85) # quality can be adjusted
        return img_byte_arr.getvalue()
    except Exception as e:
        logging.error(f"Error resizing image {image_path}: {e}")
        raise

def image_to_base64(image_bytes: bytes) -> str:
    """Converts image bytes to a base64 encoded string."""
    return base64.b64encode(image_bytes).decode('utf-8')

# --- Input Discovery & Parsing ---
def get_store_submission_paths(section_dir: Path) -> Dict[str, List[Path]]:
    """
    Returns a dict: { store_id: [list of image paths] } from a section directory.
    """
    store_submissions = {}
    for store_dir in section_dir.iterdir():
        if store_dir.is_dir():
            images = sorted([f for f in store_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            if images:
                store_id = store_dir.name  # store folder name is the store_id
                store_submissions[store_id] = images
    return store_submissions

def batch_stores(store_dict: Dict[str, List[Path]], batch_size: int) -> List[Dict[str, List[Path]]]:
    """Splits store submissions into batches of N."""
    stores = list(store_dict.items())
    return [dict(stores[i:i + batch_size]) for i in range(0, len(stores), batch_size)]


# --- Prompt Loading ---
def load_prompt(file_path: Path) -> str:
    """Loads prompt text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {file_path}")
        print(f"Error: Prompt file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading prompt {file_path}: {e}")
        raise

# --- Async Batch Evaluations ---
async def evaluate_batch_async(session, model, messages, i, max_retries=3):
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2 if model != "o4-mini" else None
    }

    for attempt in range(max_retries):
        try:
            response = await session.post(
                "https://api.openai.com/v1/chat/completions",
                json={k: v for k, v in payload.items() if v is not None},
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            return i, response.json()
        except Exception as e:
            logging.error(f"❌ Async batch {i+1} attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"❌ Async batch {i+1} attempt {attempt+1}/{max_retries} failed.")
            if attempt == max_retries - 1:
                return i, None
            await asyncio.sleep(1.5 * (attempt + 1))  # brief backoff

# --- Cost Calculation ---
def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Computes the cost of an API call based on token usage."""
    rates = MODEL_PRICING.get(model)
    if not rates:
        logging.warning(f"Pricing for model {model} not found. Cost will be 0.")
        return 0.0
    
    input_cost = (prompt_tokens / 1000) * rates["input"]
    output_cost = (completion_tokens / 1000) * rates["output"]

    return input_cost + output_cost

# --- Output Writer ---
def update_store_json(store_id: str, section: str, evaluation_content: Any, output_dir: Path):
    """Updates or creates a JSON file for the store with the evaluation results."""
    ensure_dir_exists(output_dir)
    file_path = output_dir / f"{store_id}.json"
    
    data: Dict[str, Any]
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Could not decode existing JSON for {store_id}. Creating new file.")
            data = {"store_id": store_id, "evaluations": {}}
    else:
        data = {"store_id": store_id, "evaluations": {}}

    data["evaluations"][section] = evaluation_content


    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error writing JSON for {store_id}: {e}")
        raise

def format_low_confidence_stores(store_results: dict, threshold: float = 0.75) -> str:
    """
    Formats a string of store IDs with confidence below the threshold for logging.
    """
    low_confidence_lines = []
    for store_id, result in store_results.items():
        if isinstance(result, dict):
            confidence = result.get("confidence", 1.0)
            if isinstance(confidence, (int, float)) and confidence < threshold:
                low_confidence_lines.append(f"    - Store {store_id}: Confidence {confidence:.2f}")
    
    if low_confidence_lines:
        return "  Stores requiring manual review (confidence < 0.75):\n" + "\n".join(low_confidence_lines)
    return "ALL PASS!"

### --- Create a list of message payloads — one for each batch — so they can be submitted asynchronously using evaluate_batch_async. ----
def build_messages(general_prompt, section_prompt, gold_images, batch_images):
    messages = [{"role": "system", "content": general_prompt}]
    user_content = [{"type": "text", "text": section_prompt}]

    # Attach gold standard images
    for img in gold_images:

        img_bytes = resize_image(img)
        img_b64 = image_to_base64(img_bytes)

        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    # Attach submitted images
    for store_id, store_images in batch_images.items():
        # Insert a store ID label as text before its images
        user_content.append({"type": "text", "text": f"Store ID: {store_id} — Use this exact ID as a key in your JSON output."})
        for img_path in store_images:
            img_bytes = resize_image(img_path)
            img_b64 = image_to_base64(img_bytes)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })



    messages.append({"role": "user", "content": user_content})
    return messages

# --- Main Pipeline ---
def process_section(section_name: str, submissions_dir: Path, goldstandard_dir: Path, model: str, batch_size: int, start_date: str, store_id=None):

    # Section Log initialization
    total_prompt_tokens = 0
    total_completion_tokens = 0
    all_passed_stores = []
    all_failed_stores = []

    print(f"Processing section: {section_name} (model: {model})")
    section_start_time = time.time()

    # Load prompts
    try:
        system_prompt = load_prompt(Path("testfiles/branch_audit_system_prompt.txt"))
        # only inject the start date into “YYYY-MM-DD”
        general_prompt = system_prompt.replace("YYYY-MM-DD", start_date)


        raw_section_prompt = load_prompt(Path("testfiles/branch_audit_user_prompts_all_sections") / f"{section_name}.txt")
        reinforcement = "\n\nIMPORTANT: For each store submission, you will first see a line like ‘Store ID: 045’. Use this store ID exactly when generating your JSON response."
        section_prompt = raw_section_prompt + reinforcement

    except Exception as e:
        print(f"❌ Failed to load prompt files: {e}")
        return


    # Load gold standards
    goldstandard_path = goldstandard_dir / section_name
    gold_images = sorted([f for f in goldstandard_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    if not gold_images:
        print(f"⚠️ No gold standard images found for section {section_name}")
    
    # Load submissions
    section_submissions_dir = submissions_dir / section_name
    store_dict_all = get_store_submission_paths(section_submissions_dir)

    if store_id:
        if store_id in store_dict_all:
            store_dict = {store_id: store_dict_all[store_id]}
        else:
            print(f"⚠️ Store ID '{store_id}' not found in section {section_name}")
            return
    else:
        store_dict = store_dict_all


    if not store_dict:
        print(f"⚠️ No submissions found for section {section_name}")
        return

    batches = batch_stores(store_dict, batch_size=batch_size)
    # Prepare OpenAI message payloads for each batch (for async submission)
    batch_payloads = []
    for i, batch in enumerate(batches):
        messages = build_messages(general_prompt, section_prompt, gold_images, batch)
        batch_payloads.append((i, messages, batch))

    async def process_batches():
        async with httpx.AsyncClient() as session:
            tasks = [
                evaluate_batch_async(session, model, messages, i)
                for i, messages, _ in batch_payloads
            ]
            return await asyncio.gather(*tasks)
        
    results = asyncio.run(process_batches())


    print(f"📦 Found {len(store_dict)} stores. Will process in {len(batches)} batch(es) of {batch_size}.")


    total_cost = 0.0

    for i, response_json in results:
        batch = batch_payloads[i][2]  # batch_images

        print(f"\n🌀 Processing batch {i + 1}/{len(batch_payloads)} ({len(batch)} stores)...")

        if response_json is None:
            print(f"❌ Batch {i+1} failed completely. Skipping...")
            continue

        content = response_json["choices"][0]["message"]["content"]
        usage = response_json.get("usage", {})

        # === Parsing JSON from model ===
        max_retries = 3
        for attempt in range(max_retries):
            try:
                store_results = json.loads(content.strip().strip('```json').strip('```'))
                if not isinstance(store_results, dict) or not store_results:
                    raise ValueError("Invalid response format")
                break
            except Exception as e:
                logging.error(f"❌ JSON parsing failed for batch {i+1}, attempt {attempt+1}. Error: {e}")
                print(f"❌ Batch {i+1} returned invalid JSON. Attempt {attempt+1}/{max_retries}. Skipping...")
                if attempt == max_retries - 1:
                    continue  # Skip this batch entirely

        # === Token + cost accounting ===
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        cost = compute_cost(model, prompt_tokens, completion_tokens)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        max_tokens = MODEL_TOKEN_LIMIT.get(model, 100000)
        percent_used = (total_tokens / max_tokens) * 100
        total_cost += cost
        effective_cost = max(cost, 0.01)
        is_last_batch = i == len(batch_payloads) - 1

        # === Update store JSONs ===
        batch_store_ids = list(batch.keys())  # actual store IDs used in this batch

        for output_store_id, section_result in store_results.items():
            # Match output_store_id to real store_id
            if output_store_id in batch_store_ids:
                actual_store_id = output_store_id
            elif len(store_results) == len(batch_store_ids):
                # Heuristic fallback: use order
                actual_store_id = batch_store_ids.pop(0)
            else:
                logging.warning(f"⚠️ Output store ID {output_store_id} not found in batch. Skipping.")
                continue

            update_store_json(actual_store_id, section_name, section_result, OUTPUTS_DIR)


        # === Log pass/fail/confidence ===
        failed_stores = []
        manual_review_stores = []
        num_pass = 0
        num_fail = 0
        low_conf_threshold = 0.75

        for store_id, result in store_results.items():
            if isinstance(result, dict):
                conf = result.get("confidence", 1.0)
                if isinstance(conf, (int, float)) and conf < low_conf_threshold:
                    manual_review_stores.append((store_id, conf))
                if result.get("pass") is False:
                    num_fail += 1
                    failed_stores.append(store_id)
                    all_failed_stores.append(store_id)
                elif result.get("pass") is True:
                    num_pass += 1
                    all_passed_stores.append(store_id)

        total_evaluated = num_pass + num_fail
        pass_rate = (num_pass / total_evaluated) * 100 if total_evaluated > 0 else 0.0

        low_conf_log = (
            f"  Total evaluated: {total_evaluated}\n"
            f"  Passed: {num_pass}\n"
            f"  Failed: {num_fail}\n"
            f"  Pass Rate: {pass_rate:.2f}%\n"
            f"  Stores requiring manual review (confidence < {low_conf_threshold}): {len(manual_review_stores)}\n"
        )

        if failed_stores:
            low_conf_log += "  ❌ Failed stores:\n"
            for store_id in failed_stores:
                low_conf_log += f"    - {store_id}\n"

        if manual_review_stores:
            low_conf_log += "  🧐 Manual review stores:\n"
            for store_id, conf in manual_review_stores:
                low_conf_log += f"    - {store_id}: Confidence {conf:.2f}\n"

        with open(COST_LOG_FILE, "a", encoding='utf-8') as f:
            evaluated_store_ids = list(store_results.keys())
            store_list_str = ", ".join(evaluated_store_ids)
            f.write(
                f"[{datetime.now().isoformat()}] Batch {i+1}/{len(batch_payloads)} - Model: {model}\n"
                f"  Number of Stores Evaluated: {len(batch)}\n"
                f"  Section Evaluated: {section_name}\n"
                f"  Stores Evaluated: {store_list_str}\n"
                f"  Prompt Tokens: {prompt_tokens}\n"
                f"  Completion Tokens: {completion_tokens}\n"
                f"  Total Tokens: {total_tokens}\n"
                f"  Token Usage: {percent_used:.2f}% of {max_tokens} limit\n"
                f"  Input Cost: ${(prompt_tokens / 1000) * MODEL_PRICING[model]['input']:.6f}\n"
                f"  Output Cost: ${(completion_tokens / 1000) * MODEL_PRICING[model]['output']:.6f}\n"
                f"  Estimated Cost: ${cost:.6f}\n"
                f"  Billed (Minimum) Cost: ${effective_cost:.2f}\n"
            )
            if is_last_batch:
                section_end_time = time.time()
                elapsed = section_end_time - section_start_time
                f.write(f"  Total Section Elapsed Time: {elapsed:.2f} seconds\n")
            f.write("\n" + low_conf_log + "\n")


    print(f"\n✅ Section {section_name} complete. Total cost: ${total_cost:.4f}")

    # Section Log
    section_end_time = time.time()
    elapsed = section_end_time - section_start_time
    total_stores = len(all_passed_stores) + len(all_failed_stores)
    section_log_file = LOGS_DIR / "section_log.txt"

    with open(section_log_file, "a", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"📋 Section Summary: {section_name} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
        f.write(f"  Model: {model}\n")
        f.write(f"  Stores Evaluated: {total_stores}\n")
        f.write(f"  Passed: {len(all_passed_stores)}\n")
        f.write(f"  Failed: {len(all_failed_stores)}\n")
        f.write(f"  Prompt Tokens: {total_prompt_tokens}\n")
        f.write(f"  Completion Tokens: {total_completion_tokens}\n")
        f.write(f"  Total Tokens: {total_prompt_tokens + total_completion_tokens}\n")
        f.write(f"  Estimated Cost: ${total_cost:.4f}\n")
        f.write(f"  Total Time: {elapsed:.2f} seconds\n\n")
        f.write("\n\n")

        if all_failed_stores:
            f.write("❌ Failed Stores:\n")
            for sid in all_failed_stores:
                f.write(f"  - {sid}\n")

        if all_passed_stores:
            f.write("\n✅ Passed Stores:\n")
            for sid in all_passed_stores:
                f.write(f"  - {sid}\n")

    print(f"📝 Section summary written to: {section_log_file}")

# run all at one go
async def run_single_evaluation(section, store_id, session, args):
    try:
        system_prompt_path = Path("testfiles/branch_audit_system_prompt.txt")
        section_prompt_path = Path("testfiles/branch_audit_user_prompts_all_sections") / f"{section}.txt"
        gold_image_dir = Path("testfiles/goldstandards") / section
        submission_image_dir = Path("testfiles/submissions") / section / store_id

        system_prompt = load_prompt(system_prompt_path).replace("YYYY-MM-DD", args.start_date)
        section_prompt = load_prompt(section_prompt_path) + "\n\nIMPORTANT: For each store submission, you will first see a line like ‘Store ID: 045’. Use this store ID exactly when generating your JSON response."

        gold_images = sorted([
            f for f in gold_image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        submission_images = sorted([
            f for f in submission_image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        if not submission_images:
            print(f"⚠️ No submission for store {store_id} section {section}")
            return

        messages = build_messages(
            system_prompt,
            section_prompt,
            gold_images,
            {store_id: submission_images}
        )

        max_retries = 3
        for attempt in range(max_retries):
            _, response_json = await evaluate_batch_async(session, args.model, messages, 0)
            if response_json:
                break
            else:
                print(f"⚠️ Retry {attempt+1}/{max_retries} for {store_id} / {section}")
                await asyncio.sleep(1.5 * (attempt + 1))

        if response_json:
            content = response_json["choices"][0]["message"]["content"]
            parsed = json.loads(content.strip().strip('```json').strip('```'))

            section_result = parsed.get(store_id, parsed)
            update_store_json(store_id, section, section_result, OUTPUTS_DIR)
            print(f"✅ Evaluated {store_id} / {section}")
    except Exception as e:
        print(f"❌ Error in {store_id} / {section}: {e}")

# --- Command-Line Interface ---
if __name__ == "__main__":
    setup_logging()  # Setup error logging to file
    ensure_dir_exists(LOGS_DIR)

    parser = argparse.ArgumentParser(description="Cash4You Branch Appearance Audit Automation CLI")

    parser.add_argument(
        "--ALL_IN_ONE_GO",
        action="store_true",
        help="Evaluate all sections for all stores in one async run."
    )

    parser.add_argument(
        '--section', 
        type=str,
        help="Section to evaluate (e.g., A1) or 'ALL'"
    )


    parser.add_argument(
        '--store', 
        type=str, 
        default=None, 
        help='Optional: Specify a single store ID to evaluate within the given section'
    )

    parser.add_argument(
        "--model",
        default="o4-mini",
        choices=MODEL_PRICING.keys(),
        help="OpenAI model to use for evaluation (e.g., gpt-4o, o4-mini)."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of stores to include in each API call batch."
    )

    parser.add_argument(
    "--start-date",
    type=str,
    required=True,
    help="Start date in YYYY-MM-DD format. Images earlier than this will be flagged."
    )


    args = parser.parse_args()

    if not args.ALL_IN_ONE_GO and not args.section:
        parser.error("--section is required unless --ALL_IN_ONE_GO is specified.")


    if args.section and args.section.upper() == "ALL":
        section_list = sorted([
            d.name for d in (Path("testfiles/submissions")).iterdir() if d.is_dir()
        ])
    else:
        section_list = [args.section]

    
    # Define OUTPUTS_DIR dynamically based on model
    OUTPUTS_DIR = Path("outputs") / args.model
    ensure_dir_exists(OUTPUTS_DIR)

    print("Starting Branch Appearance Audit Automation...")
    print(f"Model: {args.model}")
    if args.ALL_IN_ONE_GO:
        print("Section: ALL_IN_ONE_GO")
        print("Submission images from: testfiles/submissions/")
        print("Gold standard images from: testfiles/goldstandards/")
    else:
        print(f"Section: {args.section}")
        print(f"Submission images from: testfiles/submissions/{args.section}/")
        print(f"Gold standard images from: testfiles/goldstandards/{args.section}/")

    print(f"Output JSONs to: {OUTPUTS_DIR}")
    print(f"Cost log: {COST_LOG_FILE}")
    print(f"Error log: {ERROR_LOG_FILE}")
    print("-" * 30)

    # Use updated testfiles structure
    testfiles_dir = Path("testfiles")
    submissions_dir = testfiles_dir / "submissions"
    goldstandard_dir = testfiles_dir / "goldstandards"

    if args.ALL_IN_ONE_GO:
        print(f"🚀 Running all sections for all stores in one async run...")

        async def run_all_sections_all_stores():
            tasks = []
            async with httpx.AsyncClient() as session:
                # Dynamically gather section list from submission folder
                section_list = sorted([
                    d.name for d in (Path("testfiles/submissions")).iterdir() if d.is_dir()
                ])

                for section in section_list:
                    section_dir = submissions_dir / section
                    if not section_dir.exists():
                        print(f"⚠️ Skipping section {section} — No submission folder found.")
                        continue

                    for store_folder in section_dir.iterdir():
                        if store_folder.is_dir():
                            store_id = store_folder.name
                            tasks.append(run_single_evaluation(section, store_id, session, args))

                if not tasks:
                    print("⚠️ No evaluation tasks were created.")
                else:
                    await asyncio.gather(*tasks)

                    # === Post-evaluation integrity check ===
                    print("🔍 Checking for incomplete evaluations per store...")
                    missing_log = LOGS_DIR / "missing_sections_log.txt"
                    expected_sections = sorted([
                        d.name for d in (Path("testfiles/submissions")).iterdir() if d.is_dir()
                    ])

                    missing_report = []
                    for json_file in OUTPUTS_DIR.glob("*.json"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            evaluated_sections = list(data.get("evaluations", {}).keys())
                            store_id = data.get("store_id", json_file.stem)

                            missing_sections = sorted(set(expected_sections) - set(evaluated_sections))
                            if missing_sections:
                                missing_report.append(f"Store {store_id} is missing sections: {', '.join(missing_sections)}")

                        except Exception as e:
                            logging.error(f"⚠️ Could not read output JSON {json_file.name}: {e}")
                            continue

                    if missing_report:
                        with open(missing_log, "w", encoding="utf-8") as f:
                            f.write("\n".join(missing_report))
                        print(f"❗ Some stores are missing evaluations. See: {missing_log}")
                    else:
                        print("✅ All stores have all expected section evaluations.")
                        
                        all_end_time = time.time()
                        total_time = all_end_time - all_start_time

                        summary_log = LOGS_DIR / "all_sections_summary.txt"
                        with open(summary_log, "a", encoding="utf-8") as f:
                            f.write("="*60 + "\n")
                            f.write(f"🕒 Async Full Evaluation Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("="*60 + "\n")
                            f.write(f"  Model: {args.model}\n")
                            f.write(f"  Total Runtime: {total_time:.2f} seconds\n")
                            f.write(f"  Start Date for Evaluation: {args.start_date}\n\n")

        all_start_time = time.time()

        asyncio.run(run_all_sections_all_stores())

    else:
        for section in section_list:
            print(f"\n===== Processing Section {section} =====")
            process_section(
                section_name=section,
                submissions_dir=submissions_dir,
                goldstandard_dir=goldstandard_dir,
                model=args.model,
                batch_size=args.batch_size,
                start_date=args.start_date,
                store_id=args.store
            )



    print("-" * 30)
    print("Automation run finished.")
