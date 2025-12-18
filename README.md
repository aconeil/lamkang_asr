# ASR for Lamkang

This code base was developed to support research on developing an ASR model for Lamkang, documented in the paper titled "ASR for Language Documentation and Resource Prioritization: A Case Study from Lamkang" (unpublished currently). Additional details about running the scripts and functions are found in comments in the code.

The following details the role of each script in the repo:
* align_audio.py: automatically aligns a text transcript for an audio recording using MMS forced aligner
* compare_orthography.py: compares the text of a directory of text files with another directory of text files with the same name, producing a CER
* elan2wav.py: splits chunks of 7 annotations in ELAN into separate wav files and produces a CSV mapping the name of each file to its annotation
* folder_stats.py: calculates directory stats for lengths of audio clips. It prints the average length of a clip in a directory, the shortest clip, the longest clip and the total length of all the clips
* format_data.py: used to upload a dataset to huggingface, if desired
* gen_transcripts.py: uses an ASR model to produce transcripts for an audio clip
* train_wav2vec2.py: produces a fine-tuned version of facebook/wav2vec2-large-xlsr-53 using a specified dataset
* zeroshot.py: code to use omniASR_LLM_7B_ZS with 10 examples from Lamkang dataset

The models produced from fine-tuning facebook/wav2vec2-large-xlsr-53 can be found at https://huggingface.co/aconeil
