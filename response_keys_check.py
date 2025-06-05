import json
import logging

# Configure logging to write to mismatches.log
logging.basicConfig(
    filename='mismatches.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

# Load catalog JSON
with open('catalogg.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

# Group entries by name
entries_by_name = {}
for entry in catalog:
    name = entry.get('name')
    entries_by_name.setdefault(name, []).append(entry)

# Check duplicate names and compare response_example keys
for name, entries in entries_by_name.items():
    if len(entries) > 1:
        # Collect sets of response_example keys for each duplicate entry
        key_sets = [set(e.get('response_example', {}).keys()) for e in entries]
        reference_keys = key_sets[0]
        for idx, keys in enumerate(key_sets[1:], start=2):
            if keys != reference_keys:
                logging.info(f"Name: '{name}' has mismatched response_example keys:")
                logging.info(f"  Entry 1 keys: {sorted(reference_keys)}")
                logging.info(f"  Entry {idx} keys: {sorted(keys)}\n")

# If mismatches.log remains empty, there were no mismatches
