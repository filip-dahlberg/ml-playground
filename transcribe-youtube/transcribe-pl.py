import pytube
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Download the YouTube video using pytube
video_url = "https://www.youtube.com/watch?v=M6VWprJ8sgI"
youtube = pytube.YouTube(video_url)
audio_stream = youtube.streams.filter(only_audio=True).first()
audio_stream.download(output_path="./", filename="audio.mp3")

# Load the wav2vec2 model and tokenizer
model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-polish'
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# Load the audio file
audio_file_path = "/Users/filipdahlberg/gitrepos/ml-playground/transcribe-youtube/audio.mp3"
waveform, sample_rate = librosa.load(audio_file_path, sr=None, mono=True)

# Resample if necessary
if sample_rate != model.feature_extractor.sampling_rate:
    resampler = librosa.resample(waveform, orig_sr=sample_rate, target_sr=model.feature_extractor.sampling_rate)
    waveform = torch.FloatTensor(resampler)

# Preprocess the audio waveform
input_values = tokenizer(waveform, return_tensors='pt').input_values

# Transcribe the audio using the wav2vec2 model
with torch.no_grad():
    logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.decode(predicted_ids[0])

# Print the transcript to the console
with open("output_file.txt", "w", encoding="utf-8") as f:
    f.write(transcription)
