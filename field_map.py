import json

def update_nested_dict(obj, field_descriptions, prefix=""):
    """
    Recursively update nested dictionaries by replacing values with descriptions
    using dot notation keys from field_descriptions.
    """
    if isinstance(obj, dict):
        updated_obj = {}
        for key, value in obj.items():
            # Create the full key path using dot notation
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Always recursively handle nested dictionaries to maintain structure
                updated_obj[key] = update_nested_dict(value, field_descriptions, full_key)
            elif isinstance(value, list):
                # Handle lists (in case there are nested objects in lists)
                updated_obj[key] = [
                    update_nested_dict(item, field_descriptions, full_key) 
                    if isinstance(item, dict) else item 
                    for item in value
                ]
            else:
                # Check if we have a description for this exact key (leaf nodes only)
                if full_key in field_descriptions:
                    updated_obj[key] = field_descriptions[full_key]
                else:
                    # Keep original value if no description found
                    updated_obj[key] = value
        return updated_obj
    else:
        return obj

# Load the field descriptions mapping (JSON format)
with open('field_descriptions_cleaned.txt', 'r', encoding='utf-8') as f:
    field_descriptions = json.load(f)

# Load the original catalog JSON
with open('catalogg.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

# Update each entry's response_example by replacing values with descriptions
for entry in catalog:
    response_example = entry.get('response_example')
    if isinstance(response_example, dict):
        # Use the recursive function to handle nested structures
        entry['response_example'] = update_nested_dict(response_example, field_descriptions)

# Write the updated catalog to a new file
with open('catalogg_with_descriptions.json', 'w', encoding='utf-8') as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)

print("Updated catalog written to catalogg_with_descriptions.json")
print("Successfully mapped nested fields using dot notation keys!")