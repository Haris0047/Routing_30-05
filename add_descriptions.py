import json
import copy
import re

# Load field descriptions from field_descriptions.txt
def load_field_descriptions(filename):
    desc = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'"([^"]+)":\s*"(.+)"', line.strip())
            if match:
                key, value = match.groups()
                desc[key] = value
    return desc

def describe_response_example(example, field_desc):
    return {k: field_desc.get(k, "No description available.") for k in example}

def describe_parameters(param_list, field_desc):
    return {k: field_desc.get(k, "No description available.") for k in param_list}

with open('catalogg.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

field_desc = load_field_descriptions('field_descriptions.txt')

new_data = []
for entry in data:
    entry = copy.deepcopy(entry)
    # Add response_example_description
    if 'response_example' in entry:
        entry['response_example_description'] = describe_response_example(entry['response_example'], field_desc)
    # Add parameter descriptions
    if 'parameters' in entry:
        if 'required' in entry['parameters']:
            entry['parameters_required_description'] = describe_parameters(entry['parameters']['required'], field_desc)
        if 'optional' in entry['parameters']:
            entry['parameters_optional_description'] = describe_parameters(entry['parameters']['optional'], field_desc)
    new_data.append(entry)

with open('catalogg_with_descriptions.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print("Done! Output written to catalogg_with_descriptions.json") 