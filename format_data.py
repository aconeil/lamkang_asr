'''
This scripts uploads a dataset to huggingface from a csv that has path names
'''

import csv
import sys
import wave
import numpy
import pandas as pd
from huggingface_hub import login
from datasets import Dataset

login('INSERT_TOKEN_HERE')

numpy.set_printoptions(threshold=sys.maxsize)

def wavtoarray(file_path):
    con = wave.open(file_path)
    samples = con.getnframes()
    framerate = con.getframerate()
    audio = con.readframes(samples)
    audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)
    max_int16 = 2**15
    audio_normalised = audio_as_np_float32/max_int16
    return audio_normalised, framerate

in_data = sys.argv[1]
df= pd.read_csv(in_data)
paths = list(df['filename'])
arrays = []
framerates = []
x = 0
for path in paths:
    array, framerate = wavtoarray(path)
    arrays.append(array)
    framerates.append(framerate)
    x +=1
    print(x, "/", len(paths))
audios = []
for i in range(0, len(paths)):
    audios.append({'path':paths[i], 'array':arrays[i], 'sampling_rate':framerates[i]})

df['audio'] = audios
hf_dataset = Dataset.from_pandas(df)
hf_dataset.push_to_hub("aconeil/lmk_autoalign")
df.to_feather('train_format.feather')
