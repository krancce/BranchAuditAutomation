import os
import json
import argparse
from PIL import Image
import openai
from pathlib import Path
from typing import List, Dict, Any
import io
import base64
import logging
from datetime import datetime

# --- Configuration & Constants ---
MODEL_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015}, # per 1K tokens
    "gpt-4": {"input": 0.03, "output": 0.06},   # per 1K tokens
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}, # per 1K tokens (updated common pricing)
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006} # per 1K tokens (as per spec)
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
def get_images_from_folder(folder_path: Path) -> List[Path]:
    """Gets all JPG images from a specified folder, sorted."""
    if not folder_path.is_dir():
        logging.error(f"Input folder not found: {folder_path}")
        return []
    return sorted(folder_path.glob("*.jpg"))

def extract_store_id(filename: str) -> str:
    """Extracts store ID from image filename (e.g., 'store123_entrance_01.jpg' -> 'store123')."""
    # Assumes format like 'storeID_description.jpg' or 'storeID.jpg'
    return Path(filename).stem.lower().split("_")[0]

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
def evaluate_image(
    model: str,
    image_base64: str, # Changed from image_data: bytes
    general_prompt: str,
    section_prompt: str,
    reference_image_paths: List[Path] = None # Optional: if you want to include reference images
) -> Dict[str, Any]:
    """
    Sends image (as base64) and prompts to OpenAI Vision model.
    Optionally includes reference images.
    """
    messages = [
        {"role": "system", "content": general_prompt}
    ]

    user_content = [{"type": "text", "text": section_prompt}]
    
    # Add the primary image for evaluation
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
    })
    
    # Add reference images if provided
    # Note: The original spec's Python code plan didn't explicitly use reference_images
    # from the /reference_images/ folder in this API call. This is an extension.
    # If you have many, consider costs and token limits.
    if reference_image_paths:
        for i, ref_path in enumerate(reference_image_paths):
            try:
                ref_image_bytes = resize_image(ref_path) # Resize reference images too
                ref_image_base64 = image_to_base64(ref_image_bytes)
                user_content.insert(i+1, {"type": "text", "text": f"Reference Image {i+1} ({ref_path.name}):"}) # Add before image
                user_content.insert(i+2, {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{ref_image_base64}"}
                })
            except Exception as e:
                logging.warning(f"Could not load or process reference image {ref_path}: {e}")


    user_content.append({"type": "text", "text": "Evaluate the primary uploaded image based on the criteria and any provided reference images."})
    messages.append({"role": "user", "content": user_content})
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            # max_tokens can be set if needed, e.g., max_tokens=1000
            # response_format={ "type": "json_object" } # if expecting JSON output from model
        )
        return response.to_dict() # Convert Pydantic model to dict
    except openai.APIError as e:
        logging.error(f"OpenAI API error: {e}")
        print(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during API call: {e}")
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

# --- Main Pipeline ---
def process_section(section_name: str, input_images_dir: Path, reference_images_dir: Path, output_dir: Path, model: str):
    """Processes all images in a given section folder."""
    print(f"Processing section: {section_name} with model: {model}")
    
    general_prompt_file = Path("general_prompt.txt")
    section_prompt_file = Path(f"section_prompt_{section_name}.txt")

    if not input_images_dir.exists():
        print(f"Error: Input image directory not found: {input_images_dir}")
        logging.error(f"Input image directory not found: {input_images_dir}")
        return

    try:
        general_prompt = load_prompt(general_prompt_file)
        section_prompt = load_prompt(section_prompt_file)
    except Exception:
        return # Errors already logged by load_prompt

    images = get_images_from_folder(input_images_dir)
    if not images:
        print(f"No images found in {input_images_dir}")
        return

    total_run_cost = 0.0

    # Optional: Load reference images for the section
    # This is an interpretation of "Load reference image(s) from fixed folder"
    # The spec didn't detail how they'd be used, so I'm passing them to evaluate_image
    section_reference_images = []
    if reference_images_dir.exists():
        # Example: load reference images that start with the section name
        # e.g., entrance_sample1.jpg for 'entrance' section
        section_reference_images = sorted(reference_images_dir.glob(f"{section_name}_sample*.jpg")) 
        if section_reference_images:
            print(f"Found {len(section_reference_images)} reference image(s) for section '{section_name}'.")


    for image_path in images:
        print(f"  Processing image: {image_path.name}...")
        try:
            store_id = extract_store_id(image_path.name)
            if not store_id:
                logging.warning(f"Could not extract store ID from {image_path.name}. Skipping.")
                print(f"    Warning: Could not extract store ID from {image_path.name}. Skipping.")
                continue

            image_bytes = resize_image(image_path)
            image_base64 = image_to_base64(image_bytes)
            
            api_result = evaluate_image(
                model, 
                image_base64, 
                general_prompt, 
                section_prompt,
                reference_image_paths=section_reference_images # Pass reference images
            )
            
            # Extract usage and content
            usage = api_result.get("usage")
            content = api_result.get("choices")[0].get("message").get("content")

            if usage and content:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cost = compute_cost(model, prompt_tokens, completion_tokens)
                total_run_cost += cost
                
                update_store_json(store_id, section_name, content, output_dir)
                
                with open(COST_LOG_FILE, "a", encoding='utf-8') as f:
                    f.write(f"[{datetime.now().isoformat()}] {store_id} ({section_name}) - Model: {model} - Prompt: {prompt_tokens}, Completion: {completion_tokens} - Cost: {cost:.6f} USD\n")
                print(f"    Successfully evaluated. Cost: ${cost:.6f}")
            else:
                logging.error(f"API result missing usage or content for {image_path.name}. Result: {api_result}")
                print(f"    Error: API result malformed for {image_path.name}.")

        except Exception as e:
            logging.error(f"Failed processing {image_path.name}: {e}")
            print(f"    Error processing {image_path.name}: {e}. Check {ERROR_LOG_FILE}.")
            # Continue to next image

    print(f"\n✅ Section '{section_name}' processing complete.")
    print(f"Total API cost for this run (section '{section_name}'): ${total_run_cost:.4f}")

# --- Command-Line Interface ---
if __name__ == "__main__":
    setup_logging() # Setup error logging to file
    ensure_dir_exists(OUTPUTS_DIR)
    ensure_dir_exists(LOGS_DIR)

    parser = argparse.ArgumentParser(description="Cash4You Branch Appearance Audit Automation CLI")
    parser.add_argument(
        "--section",
        required=True,
        help="Section name to process (e.g., entrance, lobby). This corresponds to a subfolder in the input directory."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        choices=MODEL_PRICING.keys(),
        help="OpenAI model to use for evaluation."
    )
    parser.add_argument(
        "--input_dir_base",
        default="inputs",
        help="Base directory containing section-specific image folders (e.g., inputs/section_entrance)."
    )
    parser.add_argument(
        "--ref_dir",
        default="reference_images",
        help="Directory containing reference images (e.g., reference_images/entrance_sample1.jpg)."
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUTS_DIR), # Use the Path object converted to string for default
        help="Directory where JSON output files will be saved."
    )

    args = parser.parse_args()

    section_input_folder = Path(args.input_dir_base) / f"section_{args.section}"
    reference_images_folder = Path(args.ref_dir)
    output_destination_folder = Path(args.output_dir)

    # Basic check for OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        exit(1)
    
    print("Starting Branch Appearance Audit Automation...")
    print(f"Model: {args.model}")
    print(f"Input images from: {section_input_folder}")
    print(f"Reference images from: {reference_images_folder}")
    print(f"Output JSONs to: {output_destination_folder}")
    print(f"Cost log: {COST_LOG_FILE}")
    print(f"Error log: {ERROR_LOG_FILE}")
    print("-" * 30)

    process_section(
        section_name=args.section,
        input_images_dir=section_input_folder,
        reference_images_dir=reference_images_folder, # Pass reference image dir
        output_dir=output_destination_folder,
        model=args.model
    )

    print("-" * 30)
    print("Automation run finished.")
