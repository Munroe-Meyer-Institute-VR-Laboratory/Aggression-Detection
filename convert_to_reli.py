import csv
import datetime
import json
import os
import pathlib
from glob import glob


def convert_json_csv(json_files, existing_files, output_dir):
    # Get the current time to put into export files
    date = datetime.datetime.today().strftime("%B %d, %Y")
    time = datetime.datetime.now().strftime("%H:%M:%S")
    # Iterate through files
    for file in json_files:
        # If converted file exists already, skip it
        name = pathlib.Path(file).stem
        if name in existing_files:
            continue
        # Load session and split it up
        with open(file, 'r') as f:
            session = json.load(f)
        session_data = {k: v for k, v in session.items() if k in list(session.keys())[:15]}
        event_history = session["Event History"]
        # TODO: Export E4 data to CSV... somehow
        e4_data = session["E4 Data"]
        # Open output file and write session to it
        with open(os.path.join(output_dir, f"{name}.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, 'Generated on', date, time])
            # Write out the session fields
            writer.writerow(['Field', 'Value'])
            for key in session_data:
                writer.writerow([key, session[key]])
            # Write out the event history
            writer.writerow(['Tag', 'Onset', 'Offset', 'Frame', 'E4 Window'])
            for event in event_history:
                row = [event[0]]
                if type(event[1]) is list:
                    row.append(event[1][0])
                    row.append(event[1][1])
                else:
                    row.append(event[1])
                    row.append('')
                row.append(event[2])
                row.append(event[3])
                writer.writerow(row)


filepaths = [
    r"C:\GitHub\Keypoint-LSTM\experiment_results\SethSortedDataset_Binary_OVA_03_06_2022\exp_1024\inference",
    r"C:\GitHub\Keypoint-LSTM\experiment_results\SethSortedDataset_Binary_OVO_03_06_2022\exp_1024\inference"
]

for filepath in filepaths:
    output_files = [y for x in os.walk(filepath) for y in glob(os.path.join(x[0], '*.json'))]
    for output_file in output_files:
        with open(output_file) as f:
            output_json = json.load(f)
        output_json["Primary Data"] = "Reliability"
        with open(output_file, 'w') as f:
            json.dump(output_json, f)
        convert_json_csv([output_file], [], pathlib.Path(output_file).parent)
