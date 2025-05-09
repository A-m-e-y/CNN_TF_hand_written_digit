import cProfile
import pstats
from CNN_digit_recognizer import inference_model
import os


if __name__ == "__main__":

    file_path = "profile_counters.json"
    if os.path.exists(file_path):
        os.remove(file_path)

    profiler = cProfile.Profile()
    profiler.enable()

    inference_model(digit_to_find=7, run_mode='predict')  # You can change 7 to any digit you want to test

    profiler.disable()
    profiler.dump_stats("inference_profile.prof")

    # Optional: print basic stats
    stats = pstats.Stats("inference_profile.prof")
    stats.strip_dirs().sort_stats('cumtime').print_stats(20)

