import json

# 1. Load parameter→description mapping
with open('parameter_with_descriptions.txt', 'r', encoding='utf-8') as f:
    param_descriptions = json.load(f)

# 2. Load existing catalog with response_example already updated
with open('catalogg_with_descriptions.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

# 3. Transform each API entry’s parameters to key:value pairs
for entry in catalog:
    parameters = entry.get('parameters', {})

    # Build a dict for "required": { param_name: description, ... }
    required_list = parameters.get('required', [])
    described_required = {}
    for param_name in required_list:
        desc = param_descriptions.get(param_name)
        described_required[param_name] = desc if desc else ""
    entry['parameters']['required'] = described_required

    # Build a dict for "optional": { param_name: description, ... }
    optional_list = parameters.get('optional', [])
    described_optional = {}
    for param_name in optional_list:
        desc = param_descriptions.get(param_name)
        described_optional[param_name] = desc if desc else ""
    entry['parameters']['optional'] = described_optional

# 4. Write out the updated catalog
with open('catalogg_params_described.json', 'w', encoding='utf-8') as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)

print("Parameters have been converted to key:value form in catalogg_params_described.json")
