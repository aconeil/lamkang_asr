'''
This script generates a transcript of an audio file. To run:

    python gen_transcript.py audio_file.wav > audio_file.txt
'''

from datasets import load_dataset 
import csv
from transformers import pipeline
import torch
import sys
import librosa
import time

#Use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#specify the model to load
asr = pipeline("automatic-speech-recognition", device=0, model="aconeil/w2v2-lmk_augmented")

audio_file_path = sys.argv[1]

audio, sample_rate = librosa.load(audio_file_path, sr=16000)

result = asr(audio, chunk_length_s=30)# , stride_length_s=5)

print(result["text"])

