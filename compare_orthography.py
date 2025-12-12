'''
This script provides the CER of a group of text files in a directory compared to a group of text files in another directory that have the same name. To run
python directory_path_1 directory_path_2
'''

import sys
import jiwer
import os

updated = sys.argv[1]
originals = sys.argv[2]

text_count = 0
distance = 0
up_all = ""
og_all = ""
for file in os.listdir(updated):
    if file.endswith(".txt"):
        if file not in os.listdir(originals):
            print("Error.", file, " is missing from ", originals)
        else:
            og_path = originals + file
            print(og_path)
            with open(updated+file, 'r') as f1:
                up = f1.read()
                up_all = up_all + up
            with open(og_path, 'r') as f2:
                og = f2.read()
                og_all = og_all + og
            add_distance = jiwer.cer(up, og)
            distance = distance + add_distance
            text_count += 1
            print("File ", text_count, "has a CER of", add_distance)
print("Average CER:", distance/text_count)
print("Cumulative CER:",jiwer.cer(up_all, og_all))
