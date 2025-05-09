import os
import json
import time

PROFILE_FILE = "profile_counters.json"

def load_profile_data():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_profile_data(data):
    with open(PROFILE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_profile(function_name, elapsed_time):
    data = load_profile_data()
    if function_name not in data:
        data[function_name] = {"call_count": 0, "total_time": 0.0}

    data[function_name]["call_count"] += 1
    data[function_name]["total_time"] += elapsed_time

    save_profile_data(data)
