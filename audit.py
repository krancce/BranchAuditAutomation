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

# Initialize OpenAI Client
# Expects OPENAI_API_KEY environment variable to be set
def init_openai_client():
    global client
    try:
        client = openai.OpenAI()
    except openai.OpenAIError as e:
        print(f"Error initializing OpenAI client: {e}. Ensure OPENAI_API_KEY is set.")
        exit(1)

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

# --- OpenAI API Interaction ---
def evaluate_batch(model: str, general_prompt: str, section_prompt: str,
                   gold_images: List[Path], batch_images: Dict[str, List[Path]]) -> Dict[str, Any]:
    messages = [{"role": "system", "content": general_prompt}]
    user_content = [{"type": "text", "text": section_prompt}]

    global client
    # Gold standards
    for i, img_path in enumerate(gold_images):
        img_bytes = resize_image(img_path)
        img_b64 = image_to_base64(img_bytes)
        user_content.append({"type": "text", "text": f"Gold Standard {i+1}:"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

    # Add submissions
    for store_id, imgs in batch_images.items():
        user_content.append({
            "type": "text",
            "text": f"Submission from store {store_id}.\nStore ID = \"{store_id}\". When returning your JSON, use this exact store ID as the key."
        })

        for img_path in imgs:
            img_bytes = resize_image(img_path)
            img_b64 = image_to_base64(img_bytes)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

    user_content.append({"type": "text", "text": "Evaluate all submissions. Return a JSON object mapping each store ID to its audit result using the format described in the system prompt."})
    user_content.append({
        "type": "text",
        "text": (
        "Important: You must return a JSON object mapping store IDs to their audit results.\n"
        "Each key must match exactly the store ID shown in the 'Submission from store X' section.\n"
        "Do not invent, skip, reorder, or renumber store IDs.\n"
        "If you're unsure about any mapping, explicitly return a fail for that store ID with low confidence."
        )
    })


    messages.append({"role": "user", "content": user_content})

    # Build request parameters dynamically
    params = {
        "model": model,
        "messages": messages
    }

    # Only apply temperature override if the model supports it
    if model not in ["o4-mini"]:
        params["temperature"] = 0.2

    # Send request
    # Send request with error handling
    try:
        response = client.chat.completions.create(**params)
        return response.to_dict()
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI call: {e}")
        raise


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

    # The evaluation_content is likely a string from the model.
    # If it's meant to be structured JSON from the model, parse it here.
    # For now, storing as is.
    # try:
    #    data["evaluations"][section] = json.loads(evaluation_content)
    # except json.JSONDecodeError:
    #    data["evaluations"][section] = {"raw_text_evaluation": evaluation_content}

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


# --- Main Pipeline ---
def process_section(section_name: str, submissions_dir: Path, goldstandard_dir: Path, model: str, batch_size: int, start_date: str):
    print(f"Processing section: {section_name} (model: {model})")

    # Load prompts
    try:
        system_prompt = load_prompt(Path("testfiles/branch_audit_system_prompt.txt"))
        # only inject the start date into “YYYY-MM-DD”
        general_prompt = system_prompt.replace("YYYY-MM-DD", start_date)


        section_prompt = load_prompt(Path("testfiles/branch_audit_user_prompts_all_sections") / f"{section_name}.txt")
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
    store_dict = get_store_submission_paths(section_submissions_dir)

    if not store_dict:
        print(f"⚠️ No submissions found for section {section_name}")
        return

    batches = batch_stores(store_dict, batch_size=batch_size)
    print(f"📦 Found {len(store_dict)} stores. Will process in {len(batches)} batch(es) of {batch_size}.")


    total_cost = 0.0

    for i, batch in enumerate(batches):
        print(f"\n🌀 Processing batch {i + 1}/{len(batches)} ({len(batch)} stores)...")

        start_time = time.time()
        try:
            api_result = evaluate_batch(
            model=model,
            general_prompt=general_prompt,
            section_prompt=section_prompt,
            gold_images=gold_images,
            batch_images=batch
            )

            end_time = time.time()
            elapsed_time = end_time - start_time  # In seconds

            usage = api_result.get("usage", {})
            content = api_result.get("choices")[0].get("message").get("content")

            if usage and content:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = prompt_tokens + completion_tokens
                cost = compute_cost(model, prompt_tokens, completion_tokens)
                max_tokens = MODEL_TOKEN_LIMIT.get(model, 100000)  # fallback if not defined
                percent_used = (total_tokens / max_tokens) * 100
                total_cost += cost

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        store_results = json.loads(content.strip().strip('```json').strip('```'))
                        break  # Successful parse
                    except json.JSONDecodeError as e:
                        logging.error(f"❌ JSON parsing failed for batch {i+1}, attempt {attempt+1}. Error: {e}. Content:\n{content[:500]}")
                        print(f"❌ Batch {i+1} returned invalid JSON. Attempt {attempt+1} of {max_retries}. Retrying...")
                        
                        if attempt < max_retries - 1:
                            # Re-evaluate the batch to get a new response
                            time.sleep(2)  # small delay before retry
                            api_result = evaluate_batch(
                                model=model,
                                general_prompt=general_prompt,
                                section_prompt=section_prompt,
                                gold_images=gold_images,
                                batch_images=batch
                            )
                            content = api_result.get("choices")[0].get("message").get("content")
                            usage = api_result.get("usage", {})
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                            total_tokens = prompt_tokens + completion_tokens
                            cost = compute_cost(model, prompt_tokens, completion_tokens)
                            total_cost += cost
                        else:
                            effective_cost = max(cost, 0.01)
                            with open(COST_LOG_FILE, "a", encoding='utf-8') as f:
                                f.write(
                                    f"[{datetime.now().isoformat()}] Batch {i+1}/{len(batches)} - Model: {model}\n"
                                    f"  Stores: {len(batch)}\n"
                                    f"  Prompt Tokens: {prompt_tokens}\n"
                                    f"  Completion Tokens: {completion_tokens}\n"
                                    f"  Total Tokens: {total_tokens}\n"
                                    f"  Token Usage: {percent_used:.2f}% of {max_tokens} limit\n"
                                    f"  Input Cost: ${(prompt_tokens / 1000) * MODEL_PRICING[model]['input']:.6f}\n"
                                    f"  Output Cost: ${(completion_tokens / 1000) * MODEL_PRICING[model]['output']:.6f}\n"
                                    f"  Estimated Cost: ${cost:.6f}\n"
                                    f"  Billed (Minimum) Cost: ${effective_cost:.2f}\n"
                                    f"  Time Taken: {elapsed_time:.2f} seconds\n"
                                    f"  ❌ ERROR: JSON failed after {max_retries} attempts. Skipping batch.\n\n"
                                )
                            print(f"❌ Aborting batch {i+1} after {max_retries} failed attempts to get valid JSON.")
                            exit(1)



                for store_id, section_result in store_results.items():
                    update_store_json(store_id, section_name, section_result, OUTPUTS_DIR)

                effective_cost = max(cost, 0.01)  # OpenAI enforces minimum billing per request

                # Stats summary
                low_conf_threshold = 0.75

                # Build log summary string
                # Track failed store IDs and those needing manual review
                failed_stores = []
                manual_review_stores = []
                num_pass = 0
                num_fail = 0

                for store_id, result in store_results.items():
                    if isinstance(result, dict):
                        conf = result.get("confidence", 1.0)
                        if isinstance(conf, (int, float)) and conf < low_conf_threshold:
                            manual_review_stores.append((store_id, conf))
                        if result.get("pass") is False:
                            num_fail+= 1
                            failed_stores.append(store_id)
                        elif result.get("pass") is True:
                            num_pass += 1

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
                    f.write(
                        f"[{datetime.now().isoformat()}] Batch {i+1}/{len(batches)} - Model: {model}\n"
                        f"  Stores: {len(batch)}\n"
                        f"  Section Evaluated: {args.section}\n"
                        f"  Prompt Tokens: {prompt_tokens}\n"
                        f"  Completion Tokens: {completion_tokens}\n"
                        f"  Total Tokens: {total_tokens}\n"
                        f"  Token Usage: {percent_used:.2f}% of {max_tokens} limit\n"
                        f"  Input Cost: ${(prompt_tokens / 1000) * MODEL_PRICING[model]['input']:.6f}\n"
                        f"  Output Cost: ${(completion_tokens / 1000) * MODEL_PRICING[model]['output']:.6f}\n"
                        f"  Estimated Cost: ${cost:.6f}\n"
                        f"  Billed (Minimum) Cost: ${effective_cost:.2f}\n"
                        f"  Time Taken: {elapsed_time:.2f} seconds\n\n"
                        f"{low_conf_log}\n"
                    )

            else:
                logging.warning(f"API result malformed in batch {i+1}. No content or usage returned.")

        except Exception as e:
            print(f"❌ Error processing batch {i+1}: {e}")
            logging.error(f"Batch {i+1} failed: {e}")

    print(f"\n✅ Section {section_name} complete. Total cost: ${total_cost:.4f}")

# --- Command-Line Interface ---
if __name__ == "__main__":
    setup_logging()  # Setup error logging to file
    ensure_dir_exists(LOGS_DIR)

    parser = argparse.ArgumentParser(description="Cash4You Branch Appearance Audit Automation CLI")

    parser.add_argument(
        "--section",
        required=True,
        help="Section name to process (e.g., A1, B2, etc.). This must match the folder names in testfiles/submissions and testfiles/goldstandards."
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

    # Initialize the openAI client
    init_openai_client()
    
    # Define OUTPUTS_DIR dynamically based on model
    OUTPUTS_DIR = Path("outputs") / args.model
    ensure_dir_exists(OUTPUTS_DIR)

    print("Starting Branch Appearance Audit Automation...")
    print(f"Model: {args.model}")
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

    process_section(
        section_name=args.section,
        submissions_dir=submissions_dir,
        goldstandard_dir=goldstandard_dir,
        model=args.model,
        batch_size=args.batch_size,
        start_date=args.start_date
    )

    print("-" * 30)
    print("Automation run finished.")
