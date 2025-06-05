import json
import os

# Paths (set these as needed)
CATALOG_PATH = 'catalogg.json'
FIELD_DESC_PATH = 'field_descriptions_cleaned.txt'

def load_catalog_fields(catalog_path):
    """Extract all unique field names from all response_example objects in catalogg.json, including nested."""
    def extract_fields(obj, prefix=''):
        fields = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    fields |= extract_fields(v, prefix=prefix + k + '.')
                else:
                    fields.add(prefix + k)
        elif isinstance(obj, list):
            for item in obj:
                fields |= extract_fields(item, prefix=prefix)
        return fields

    with open(catalog_path, encoding='utf-8') as f:
        catalog = json.load(f)

    all_fields = set()
    for entry in catalog:
        resp = entry.get('response_example')
        if resp:
            all_fields |= extract_fields(resp)
    return all_fields

def load_described_fields(field_desc_path):
    """Load described field names from your .txt file (expects JSON dict)."""
    with open(field_desc_path, encoding='utf-8') as f:
        field_desc = json.load(f)
    return set(field_desc.keys())

def main():
    catalog_fields = load_catalog_fields(CATALOG_PATH)
    described_fields = load_described_fields(FIELD_DESC_PATH)

    # If your descriptions file includes some fields with dot notation (nested), match strictly!
    missing = sorted([field for field in catalog_fields if field not in described_fields])

    print(f"Total unique fields in catalogg.json: {len(catalog_fields)}")
    print(f"Total fields described: {len(described_fields)}")
    print(f"Missing fields ({len(missing)}):")
    for field in missing:
        print(f'  "{field}",')

    # Optional: write to file for tracking
    with open('missing_fields.txt', 'w', encoding='utf-8') as out:
        for field in missing:
            out.write(field + "\n")

if __name__ == "__main__":
    main()
