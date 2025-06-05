import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

input_file = "field_descriptions.txt"
output_file = "field_descriptions_llm.txt"

def get_llm_description(field, example):
    prompt = (
        f"Field: {field}\n"
        f"Example value: {example}\n"
        "Write a detailed, human-friendly description for this field, including what it means, its type, and an example. "
        "Format: description with all details combined. (String)"
    )
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # or "gpt-4.1" if that's your endpoint
        messages=[
            {"role": "system", "content": "You are a helpful technical documentation assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=120,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        if ":" not in line:
            continue
        field, example = line.split(":", 1)
        field = field.strip().strip('"')
        example = example.strip().strip('",\n')
        try:
            desc = get_llm_description(field, example)
            fout.write(f'"{field}": "{desc}"\n')
            print(f"Done: {field}")
        except Exception as e:
            print(f"Error for {field}: {e}")
            fout.write(f'"{field}": "Description generation failed."\n')

print(f"All done! Output written to {output_file}") 