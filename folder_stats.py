import os
import wave
import sys

folder_path = sys.argv[1]

def get_dur(folder):
    total_files = 0
    total_length = 0
    shortest = 3000000
    longest = 0
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            total_files += 1
            filepath = os.path.join(folder_path, file)
            with wave.open(filepath, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration_s = frames / float(rate)
                total_length += duration_s
                if duration_s < shortest:
                    shortest = duration_s
                if duration_s > longest:
                    longest = duration_s
    return total_length, total_files, shortest, longest

total_length, total_files, shortest, longest = get_dur(folder_path)
print(total_length/total_files, shortest, longest)
