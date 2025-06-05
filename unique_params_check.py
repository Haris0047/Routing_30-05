import json

# 1. Load unique_parameters.json
with open('unique_parameters.json', 'r', encoding='utf-8') as f:
    unique_params = json.load(f)

# Determine the set of valid parameter names
if isinstance(unique_params, dict):
    unique_set = set(unique_params.keys())
elif isinstance(unique_params, list):
    unique_set = set(unique_params)
else:
    raise ValueError("unique_parameters.json must contain either a JSON object or array")

# 2. Load catalogg_with_descriptions.json
with open('catalogg_with_descriptions.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

# 3. Extract all parameter names used in the catalog
catalog_params = set()
for entry in catalog:
    parameters = entry.get('parameters', {})

    # 'required' may be either a dict (param_name → description) or a list of names
    required = parameters.get('required', {})
    if isinstance(required, dict):
        catalog_params.update(required.keys())
    elif isinstance(required, list):
        catalog_params.update(required)

    # 'optional' similarly could be a dict or a list
    optional = parameters.get('optional', {})
    if isinstance(optional, dict):
        catalog_params.update(optional.keys())
    elif isinstance(optional, list):
        catalog_params.update(optional)

# 4. Compute parameters missing from unique_parameters.json
missing_params = catalog_params - unique_set

# 5. Print the results
print("Parameters present in catalogg_with_descriptions.json but missing from unique_parameters.json:")
print(json.dumps(sorted(missing_params), indent=2))
