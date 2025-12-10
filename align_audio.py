#This code is based on the following tutorial: https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html

#To run: python3 align_audio.py directory_name

import sys
import os
import torch
import torchaudio
from typing import List
import matplotlib.pyplot as plt
from torchaudio.pipelines import MMS_FA as bundle
import re
from pympi.Elan import Eaf
import librosa
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = bundle.get_model()
model.to(device)

tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

directory = sys.argv[1]

def resample_and_save(filename):
    waveform, sr_org = librosa.load(f"{directory}{filename}.wav", sr=None)
    resampled = librosa.resample(waveform, orig_sr=sr_org, target_sr=16000)
    sf.write(f"{directory}{filename}_resampled.wav", resampled, 16000)

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans

def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z'])", " ", text)
    text = re.sub(' +', ' ', text)
    print(text)
    return text.strip()

def alignment_list(waveform, token_spans, num_frames, transcript, name):
    eaf = Eaf()
    eaf.add_tier("default")
    eaf.add_linked_file(f"{directory}{name}_resampled.wav")
    for i in range(0, len(token_spans)):
        spans = token_spans[i]
        ratio = waveform.size(1) / num_frames
        start = ((ratio * spans[0].start)/16000) * 1000
        end = ((ratio * spans[-1].end)/16000) * 1000
        print(transcript[i], start, end)
        eaf.add_annotation("default", int(start), int(end), transcript[i])
    eaf.to_file(f"{directory}{name}.eaf")
    return


def align_files(directory):
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            name = file.replace(".txt", "")
            resample_and_save(name)
            waveform, sample_rate = torchaudio.load(f"{directory}{name}_resampled.wav")
            assert sample_rate == bundle.sample_rate
            with open(f"{directory}{file}", "r") as f:
                text_normalized = ""
                for line in f:
                    text_normalized += " " + normalize_uroman(line)
            transcript = text_normalized.split() 
            tokens = tokenizer(transcript)
            emission, token_spans = compute_alignments(waveform, transcript)
            num_frames = emission.size(1)
            alignment_list(waveform, token_spans, num_frames, transcript, name)           

align_files(directory)
