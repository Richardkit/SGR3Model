"""Legacy module name retained solely for the original JSON parser; no HTTP client."""
import json

def parse_json(text):
    text = text.strip().removesuffix('<END_JSON>').strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)
