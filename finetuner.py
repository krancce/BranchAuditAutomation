import os
import json
from collections import defaultdict

RESULTS_DIR = os.path.join('outputs', 'o4-mini')
LOG_PATH = os.path.join('logs','FineTuneResultLog.txt')

fail_count = defaultdict(int)
all_sections = set()
store_missing_sections = {}
timestamp_fail_count = defaultdict(int)

# First pass: collect all section codes present in any file
for filename in os.listdir(RESULTS_DIR):
    if filename.startswith('store') and filename.endswith('.json'):
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_sections.update(data.keys())
        except Exception as e:
            print(f"Error reading {filename}: {e}")

all_sections = sorted(all_sections)

# Second pass: tally fails, record incomplete stores
store_count = 0
for filename in os.listdir(RESULTS_DIR):
    if filename.startswith('store') and filename.endswith('.json'):
        filepath = os.path.join(RESULTS_DIR, filename)
        store_num = filename[5:8]  # extract '001' from 'store001.json'
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            store_missing_sections[store_num] = f"Could not parse JSON ({e})"
            continue
        store_count += 1

        # Count fails for present sections
        for section in data:
            if section in all_sections and not data[section].get('pass', True):
                fail_count[section] += 1
                # Check for 'timestamp' in failed_criteria for timestamp-related fails
                failed_criteria = data[section].get('failed_criteria', [])
                if any('timestamp' in crit.lower() for crit in failed_criteria):
                    timestamp_fail_count[section] += 1


        # Identify missing sections for this store
        missing = [s for s in all_sections if s not in data]
        if missing:
            store_missing_sections[store_num] = missing

# Output to log file
with open(LOG_PATH, 'w', encoding='utf-8') as log:
    log.write(f"Total stores processed: {store_count}\n\n")
    log.write(f"{'Section':<8} {'Failed Stores':<14} {'Timestamp':<12} {'Failed %':<10} {'Non-Timestamp %':<16}\n")
    log.write('-' * 68 + '\n')
    for section in all_sections:
        failed = fail_count.get(section, 0)
        timestamp_failed = timestamp_fail_count.get(section, 0)
        percent = (failed / store_count * 100) if store_count > 0 else 0
        non_timestamp_percent = ((failed - timestamp_failed) / store_count * 100) if store_count > 0 else 0
        log.write(f"{section:<8} {failed:<14} {timestamp_failed:<12} {percent:>6.1f}%      {non_timestamp_percent:>6.1f}%\n")
    log.write('\nStores with missing/incomplete sections:\n')
    log.write('-' * 68 + '\n')
    if not store_missing_sections:
        log.write("All stores had all detected sections.\n")
    else:
        for store_num, missing in store_missing_sections.items():
            if isinstance(missing, list):
                missing_str = ', '.join(missing)
                log.write(f"Store {store_num}: missing {missing_str}\n")
            else:
                log.write(f"Store {store_num}: {missing}\n")

print(f"Summary written to {LOG_PATH}")

# === BEGIN PER-SECTION FAILURE EXPORT ===

# 1. Create output folder for section failure summaries (project root)
failure_summary_dir = os.path.join(os.getcwd(), "Failure Summary")
os.makedirs(failure_summary_dir, exist_ok=True)

# 2. Prepare dict to collect failures for each section (not due to timestamp)
section_failures = {section: [] for section in all_sections}

# 3. Scan all results again to fill per-section failure files
for filename in os.listdir(RESULTS_DIR):
    if filename.startswith('store') and filename.endswith('.json'):
        filepath = os.path.join(RESULTS_DIR, filename)
        store_num = filename[5:8]  # e.g., '001'
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue  # skip broken JSON

        for section, eval_data in data.items():
            # Only failed, only if not due to timestamp
            if not eval_data.get('pass', True):
                failed_criteria = eval_data.get('failed_criteria', [])
                if any('timestamp' in crit.lower() for crit in failed_criteria):
                    continue  # skip: failed due to timestamp
                # Export only if this section is tracked
                if section in section_failures:
                    section_failures[section].append({
                        "store": store_num,
                        "failed_criteria": failed_criteria,
                        "action_plan": eval_data.get('action_plan', []),
                        "visual_feedback": eval_data.get('visual_feedback', []),
                        "notes": eval_data.get('notes', "")
                    })

# 4. Write out JSON files for each section (A1.json, etc. with case match)
for section in all_sections:
    output_path = os.path.join(failure_summary_dir, f"{section}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(section_failures[section], f, ensure_ascii=False, indent=2)

print(f"Per-section non-timestamp failures exported to: {failure_summary_dir}")
