import json
import os
import re
from collections import OrderedDict

# Paths
FIELD_DESC_PATH = 'field_descriptions_with_descriptions.txt'
OUTPUT_PATH = 'field_descriptions_cleaned.txt'

def parse_json_with_duplicates(file_path):
    """
    Parse JSON file that may contain duplicate keys by processing it as text.
    Returns OrderedDict with only first occurrence of each key.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Remove outer braces and whitespace
        if content.startswith('{') and content.endswith('}'):
            content = content[1:-1].strip()
        else:
            raise ValueError("File doesn't appear to be a JSON object")
        
        # Split by lines and process each key-value pair
        lines = content.split('\n')
        
        result = OrderedDict()
        current_key = None
        current_value_lines = []
        in_value = False
        brace_count = 0
        
        for line in lines:
            line = line.strip()
            if not line or line == ',':
                continue
            
            # Check if this line starts a new key-value pair
            key_match = re.match(r'^"([^"]+)"\s*:\s*(.*)$', line)
            
            if key_match and not in_value:
                # Save previous key-value if exists
                if current_key is not None:
                    value_str = ' '.join(current_value_lines).rstrip(',')
                    if current_key not in result:
                        try:
                            result[current_key] = json.loads(value_str)
                        except:
                            result[current_key] = value_str.strip('"')
                    else:
                        print(f"Removed duplicate key: '{current_key}'")
                
                # Start new key-value pair
                current_key = key_match.group(1)
                value_part = key_match.group(2)
                current_value_lines = [value_part]
                
                # Check if value is complete on same line
                if value_part.endswith(','):
                    in_value = False
                elif value_part.startswith('"') and value_part.endswith('"') and value_part.count('"') == 2:
                    in_value = False
                elif value_part in ['true', 'false', 'null'] or re.match(r'^-?\d+(\.\d+)?$', value_part.rstrip(',')):
                    in_value = False
                else:
                    in_value = True
                    brace_count = value_part.count('{') - value_part.count('}')
                    brace_count += value_part.count('[') - value_part.count(']')
            
            elif in_value:
                # Continue collecting value lines
                current_value_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                brace_count += line.count('[') - line.count(']')
                
                if brace_count <= 0 and (line.endswith(',') or line.endswith('}') or line.endswith(']')):
                    in_value = False
        
        # Handle last key-value pair
        if current_key is not None:
            value_str = ' '.join(current_value_lines).rstrip(',')
            if current_key not in result:
                try:
                    result[current_key] = json.loads(value_str)
                except:
                    result[current_key] = value_str.strip('"')
            else:
                print(f"Removed duplicate key: '{current_key}'")
        
        return result
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

def simple_json_parser(file_path):
    """
    Alternative simpler approach using regex to find all key-value pairs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all key-value pairs using regex
        # This pattern looks for "key": followed by any JSON value
        pattern = r'"([^"]+)"\s*:\s*(?:"(?:[^"\\]|\\.)*"|[^,}]+|{[^}]*}|\[[^\]]*\])'
        matches = re.findall(pattern, content, re.DOTALL)
        
        result = OrderedDict()
        duplicate_count = 0
        
        # Re-parse to get full key-value pairs
        key_value_pattern = r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|[^,}\n]+)'
        
        for match in re.finditer(key_value_pattern, content):
            key = match.group(1)
            value_str = match.group(2).strip()
            
            if key not in result:
                # Try to parse the value as JSON
                try:
                    if value_str.startswith('"') and value_str.endswith('"'):
                        # String value
                        result[key] = json.loads(value_str)
                    elif value_str.lower() in ['true', 'false']:
                        result[key] = json.loads(value_str.lower())
                    elif value_str.lower() == 'null':
                        result[key] = None
                    elif re.match(r'^-?\d+(\.\d+)?$', value_str):
                        result[key] = json.loads(value_str)
                    else:
                        result[key] = value_str.strip('"')
                except:
                    result[key] = value_str.strip('"')
            else:
                duplicate_count += 1
                print(f"Removed duplicate key: '{key}'")
        
        print(f"Found {duplicate_count} duplicate keys")
        return result
        
    except Exception as e:
        print(f"Error in simple parser: {e}")
        return None

def remove_duplicate_keys(input_path, output_path):
    """
    Remove duplicate keys from JSON file with duplicates.
    """
    print("Attempting to parse JSON file with duplicate keys...")
    
    # Try the simple parser first
    cleaned_data = simple_json_parser(input_path)
    
    if cleaned_data is None:
        print("Simple parser failed, trying advanced parser...")
        cleaned_data = parse_json_with_duplicates(input_path)
    
    if cleaned_data is None:
        print("Both parsers failed. Please check your file format.")
        return False
    
    try:
        # Write the cleaned data to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dict(cleaned_data), f, indent=2, ensure_ascii=False)
        
        print(f"\n--- Summary ---")
        print(f"Unique keys found: {len(cleaned_data)}")
        print(f"Cleaned file saved as: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

def backup_original(file_path):
    """Create a backup of the original file."""
    if os.path.exists(file_path):
        backup_path = file_path + '.backup'
        try:
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            print(f"Backup created: {backup_path}")
            return True
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
            return False
    return False

def replace_original_option():
    """Ask user if they want to replace the original file."""
    while True:
        choice = input("\nDo you want to replace the original file? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def main():
    print("=== Duplicate Key Remover (for Invalid JSON) ===")
    print(f"Processing: {FIELD_DESC_PATH}")
    
    # Check if input file exists
    if not os.path.exists(FIELD_DESC_PATH):
        print(f"Error: Input file '{FIELD_DESC_PATH}' not found.")
        return
    
    # Create backup
    backup_created = backup_original(FIELD_DESC_PATH)
    
    # Remove duplicates
    success = remove_duplicate_keys(FIELD_DESC_PATH, OUTPUT_PATH)
    
    if success:
        # Ask if user wants to replace original
        if replace_original_option():
            try:
                # Replace original with cleaned version
                os.replace(OUTPUT_PATH, FIELD_DESC_PATH)
                print(f"Original file '{FIELD_DESC_PATH}' has been updated.")
                if backup_created:
                    print(f"Original backed up as '{FIELD_DESC_PATH}.backup'")
            except Exception as e:
                print(f"Error replacing original file: {e}")
        else:
            print(f"Cleaned file saved as '{OUTPUT_PATH}'")
            print(f"Original file '{FIELD_DESC_PATH}' unchanged.")

if __name__ == "__main__":
    main()