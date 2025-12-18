from omnilingual_asr.models.inference.pipeline import (ASRInferencePipeline, ContextExample)

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B_ZS")

context_examples = [ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk200705_001_clip11.wav", "krthoon lam da, mthungbi ngu snung ki mdo hur theihaai mkheng avaa dulti tha lei pthii lam da"),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk2009_003_clip36.wav", "Ah, Delhi thung tha kryun naa kum knii mphaak dok ka, ah, adil ki kum knii mphaak ka Delhi thung tha k'am thung ngi"),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk200805_001_clip8.wav", "m'am laan ni Benglam va ngu mrii lou da ding bul leen psuk thi da ngu mKhungsaai rek ka vang lam da Hoi ya hoi ya da mtxen in mtxen in mtxen in"),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk2009_002_clip7.wav", "mrvan knlet thung bi snung ki mnao kdi you mdo ptlaak pthii'a lam mi kploom daat lam mi mdo arvan phaak pi di yu"),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk2009_004_clip2.wav", "ching loon, loon tha ding nga knem mi moong me, noom lu kmoong, ding lung nga am paa me ava thung thang ngi ding bul khat kmoong pii To'a, ava ding bul la ngi naaspati ding bul hep pa,"),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk20080222_001_clip60.wav", "naa amaai txhuum mi pnuu krvee chaak ka, nei yi naa achaa paa rang nga krhui chmaang,” pdainu"),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk200805_001_clip13.wav", "Benglam ee you pchaang lam mdau mditto."),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk200705_001_clip1.wav", "Nii khat va mii paa khat ta theihaai klin paa bul khat kchei arlei don da, hang kal da"),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk2009_004_clip65.wav", "tupii hei pii da au thung ki ava nao va tupii khum da anto cycle thou da anto kvang cha,"),
	ContextExample("/N/slate/aconeil/lamkang_audio/aligned/updated/manual_align/clips/train/lmk2009_003_clip21.wav", "pastor tloo da k'am pdainu tun va, pastor maa pthii txha anto in nong anto suu da k'am.")
]

transcription_0 = pipeline.transcribe_with_context(["test_files/output_000.wav"], context_examples=[context_examples], batch_size=1)
transcription_1 = pipeline.transcribe_with_context(["test_files/output_001.wav"], context_examples=[context_examples], batch_size=1)
transcription_2 = pipeline.transcribe_with_context(["test_files/output_002.wav"], context_examples=[context_examples], batch_size=1)
transcription_3 = pipeline.transcribe_with_context(["test_files/output_003.wav"], context_examples=[context_examples], batch_size=1)

print(transcription_0[0], transcription_1[0], transcription_2[0], transcription_3[0])
