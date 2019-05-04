import json

def get_cat_to_name(filename = 'cat_to_name.json'):
    with open(filename, 'r') as f:
        return json.load(f)
