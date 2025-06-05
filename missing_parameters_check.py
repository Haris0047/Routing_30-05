import json
import os

# Paths
CATALOG_PATH = 'catalogg.json'
UNIQUE_PARAMS_PATH = 'unique_parameters.json'

def load_catalog_parameters(catalog_path):
    """Extract all unique parameter names from all entries in catalogg.json."""
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = json.load(f)
        
        all_required_params = set()
        all_optional_params = set()
        all_params = set()
        
        for entry in catalog:
            # Get parameters from each entry
            parameters = entry.get('parameters', {})
            
            # Extract required parameters
            required = parameters.get('required', [])
            if isinstance(required, list):
                all_required_params.update(required)
                all_params.update(required)
            
            # Extract optional parameters
            optional = parameters.get('optional', [])
            if isinstance(optional, list):
                all_optional_params.update(optional)
                all_params.update(optional)
        
        return all_params, all_required_params, all_optional_params
        
    except FileNotFoundError:
        print(f"Error: File '{catalog_path}' not found.")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{catalog_path}': {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return None, None, None

def load_unique_parameters(unique_params_path):
    """Load the unique parameters from unique_parameters.json."""
    try:
        with open(unique_params_path, 'r', encoding='utf-8') as f:
            unique_params = json.load(f)
        
        # Extract parameter names (keys)
        if isinstance(unique_params, dict):
            return set(unique_params.keys())
        else:
            print(f"Error: Expected dictionary in '{unique_params_path}', got {type(unique_params).__name__}")
            return None
            
    except FileNotFoundError:
        print(f"Error: File '{unique_params_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{unique_params_path}': {e}")
        return None
    except Exception as e:
        print(f"Error loading unique parameters: {e}")
        return None

def compare_parameters(catalog_params, catalog_required, catalog_optional, unique_params):
    """Compare parameters and find missing ones."""
    
    # Find parameters that are in unique_parameters.json but not in catalogg.json
    missing_from_catalog = sorted([param for param in unique_params if param not in catalog_params])
    
    # Find parameters that are in catalogg.json but not in unique_parameters.json
    missing_from_unique = sorted([param for param in catalog_params if param not in unique_params])
    
    return missing_from_catalog, missing_from_unique

def analyze_parameter_usage(catalog_path, unique_params):
    """Analyze how parameters are used across different entries."""
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = json.load(f)
        
        param_usage = {}
        
        for entry in catalog:
            entry_name = entry.get('name', 'Unknown')
            parameters = entry.get('parameters', {})
            
            required = parameters.get('required', [])
            optional = parameters.get('optional', [])
            
            for param in required + optional:
                if param not in param_usage:
                    param_usage[param] = {
                        'count': 0,
                        'required_in': [],
                        'optional_in': [],
                        'in_unique_params': param in unique_params
                    }
                
                param_usage[param]['count'] += 1
                
                if param in required:
                    param_usage[param]['required_in'].append(entry_name)
                if param in optional:
                    param_usage[param]['optional_in'].append(entry_name)
        
        return param_usage
        
    except Exception as e:
        print(f"Error analyzing parameter usage: {e}")
        return None

def main():
    print("=== Parameter Comparison Analysis ===")
    
    # Check if files exist
    if not os.path.exists(CATALOG_PATH):
        print(f"Error: '{CATALOG_PATH}' not found.")
        return
    
    if not os.path.exists(UNIQUE_PARAMS_PATH):
        print(f"Error: '{UNIQUE_PARAMS_PATH}' not found.")
        return
    
    # Load data
    print("Loading catalog parameters...")
    catalog_params, catalog_required, catalog_optional = load_catalog_parameters(CATALOG_PATH)
    
    if catalog_params is None:
        return
    
    print("Loading unique parameters...")
    unique_params = load_unique_parameters(UNIQUE_PARAMS_PATH)
    
    if unique_params is None:
        return
    
    # Compare parameters
    missing_from_catalog, missing_from_unique = compare_parameters(
        catalog_params, catalog_required, catalog_optional, unique_params
    )
    
    # Print summary
    print(f"\n--- SUMMARY ---")
    print(f"Total parameters in catalogg.json: {len(catalog_params)}")
    print(f"  - Required parameters: {len(catalog_required)}")
    print(f"  - Optional parameters: {len(catalog_optional)}")
    print(f"Total parameters in unique_parameters.json: {len(unique_params)}")
    
    # Missing from catalog
    print(f"\n--- PARAMETERS MISSING FROM CATALOGG.JSON ---")
    print(f"Count: {len(missing_from_catalog)}")
    if missing_from_catalog:
        print("These parameters are defined in unique_parameters.json but not used in any catalog entry:")
        for param in missing_from_catalog:
            print(f"  - {param}")
    else:
        print("✅ All parameters from unique_parameters.json are used in catalogg.json")
    
    # Missing from unique parameters
    print(f"\n--- PARAMETERS MISSING FROM UNIQUE_PARAMETERS.JSON ---")
    print(f"Count: {len(missing_from_unique)}")
    if missing_from_unique:
        print("These parameters are used in catalogg.json but not defined in unique_parameters.json:")
        for param in missing_from_unique:
            print(f"  - {param}")
    else:
        print("✅ All catalog parameters are defined in unique_parameters.json")
    
    # Detailed parameter usage analysis
    print(f"\n--- PARAMETER USAGE ANALYSIS ---")
    param_usage = analyze_parameter_usage(CATALOG_PATH, unique_params)
    
    if param_usage:
        # Parameters not in unique_parameters.json
        undefined_params = [param for param, info in param_usage.items() if not info['in_unique_params']]
        
        if undefined_params:
            print(f"\nParameters used in catalog but not defined in unique_parameters.json:")
            for param in sorted(undefined_params):
                info = param_usage[param]
                print(f"  - '{param}' (used {info['count']} times)")
                if info['required_in']:
                    print(f"    Required in: {', '.join(info['required_in'][:3])}{'...' if len(info['required_in']) > 3 else ''}")
                if info['optional_in']:
                    print(f"    Optional in: {', '.join(info['optional_in'][:3])}{'...' if len(info['optional_in']) > 3 else ''}")
    
    # Write missing parameters to files for easy reference
    if missing_from_catalog:
        with open('missing_from_catalog.txt', 'w', encoding='utf-8') as f:
            for param in missing_from_catalog:
                f.write(param + '\n')
        print(f"\n📝 Missing from catalog written to: missing_from_catalog.txt")
    
    if missing_from_unique:
        with open('missing_from_unique_params.txt', 'w', encoding='utf-8') as f:
            for param in missing_from_unique:
                f.write(param + '\n')
        print(f"📝 Missing from unique_parameters.json written to: missing_from_unique_params.txt")

if __name__ == "__main__":
    main()