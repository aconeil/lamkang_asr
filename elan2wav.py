#This file expects elan files that have one tier named "default"
#To run: python3 elan2wav.py directory_name
from speach import elan
import csv
import sys
import os

directory = sys.argv[1] 

with open(f"{directory}clips/transcript_map.csv", "w", newline= "") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Annotation", "Filename"])
    for elan_file in os.listdir(directory):
        if not elan_file.endswith(".eaf"):
            continue
        eaf = elan.read_eaf(f"{directory}{elan_file}")
        for tier in eaf:
            for idx, ann in enumerate(eaf["default"], start=1):
                short_name = elan_file.replace(".eaf","")
                f_name = f"{directory}clips/{short_name}_clip{idx}.wav"
                eaf.cut(ann, f_name)
                csv_writer.writerow([ann, f_name])
