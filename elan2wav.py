#This file expects elan files that have one tier named "default"
#To run: python3 elan2wav.py directory_name
from speach import elan
import csv
import sys
import os

directory = sys.argv[1] 

def chunk_list(input_list, ngrams):
    for i in range(0, len(input_list), ngrams):
        if len(input_list)-i > 13:
            yield input_list[i:i + ngrams]
        elif len(input_list) - i < 7:
            yield
        else:
            yield input_list[i:]

with open(f"{directory}clips/train/metadata.csv", "w", newline= "") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["transcription", "file_name"])
    for elan_file in os.listdir(directory):
        if not elan_file.endswith(".eaf"):
            continue
        print(elan_file)
        eaf = elan.read_eaf(f"{directory}{elan_file}")
        annotations = list(eaf["default"])
        print(annotations)
        short_name = elan_file.replace(".eaf","")
        for idx, chunk in enumerate(chunk_list(annotations, 7), start=1):
            if not chunk:
                continue
            combined = " ".join([ann.text for ann in chunk])
            start_ts = chunk[0].from_ts
            end_ts = chunk[-1].to_ts
            combined_ann = elan.Annotation(from_ts=start_ts, to_ts=end_ts, value=combined, ID=idx)
            f_path = f"{directory}clips/train/{short_name}_clip{idx}.wav"
            f_name = f"{short_name}_clip{idx}.wav"
            eaf.cut(combined_ann, f_path)
            csv_writer.writerow([combined_ann, f_name])
