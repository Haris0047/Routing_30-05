import json

def extract_names_from_catalog(catalog_file="catalogg_params_described_simplified.json", output_file="company_names.txt"):
    """
    Extract company names from catalog.json and save them to a text file.
    
    Args:
        catalog_file: Path to the catalog.json file
        output_file: Path to the output text file for company names
    """
    try:
        # Read the catalog.json file
        with open(catalog_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        company_names = []
        
        # Extract names from each entry
        if isinstance(data, list):
            # If data is a list of entries
            for entry in data:
                if isinstance(entry, dict) and 'name' in entry:
                    name = entry['name']
                    if name and isinstance(name, str):
                        company_names.append(name.strip())
        elif isinstance(data, dict):
            # If data is a single entry
            if 'name' in data:
                name = data['name']
                if name and isinstance(name, str):
                    company_names.append(name.strip())
        
        # Remove duplicates while preserving order
        unique_names = []
        seen = set()
        for name in company_names:
            if name not in seen:
                unique_names.append(name)
                seen.add(name)
        
        # Save names to file
        with open(output_file, 'w', encoding='utf-8') as file:
            for name in unique_names:
                file.write(name + '\n')
        
        print(f"Successfully extracted {len(unique_names)} company names from {catalog_file}")
        print(f"Names saved to {output_file}")
        
        # Print first few names as preview
        if unique_names:
            print("\nFirst few company names:")
            for i, name in enumerate(unique_names[:5]):
                print(f"  {i+1}. {name}")
            if len(unique_names) > 5:
                print(f"  ... and {len(unique_names) - 5} more")
    
    except FileNotFoundError:
        print(f"Error: {catalog_file} not found. Please make sure the file exists in the current directory.")
    except json.JSONDecodeError:
        print(f"Error: {catalog_file} contains invalid JSON format.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run the extraction
    extract_names_from_catalog()