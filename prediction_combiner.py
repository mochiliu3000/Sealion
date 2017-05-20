import re

def parse_prediction(log_path):
    # /Users/ibm/GitRepo/darknet/data/0_8.JPEG: Predicted in 40.548927 seconds.

    start_line_regex = re.compile("[-~]+:\\s+Predicted in ([-+]?[0-9]*\.?[0-9]*) seconds\.")
    with open(log_path, "r") as log:
        for line in log:
            search_start = start_line_regex.search(line)

