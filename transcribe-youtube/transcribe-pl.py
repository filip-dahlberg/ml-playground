import pytube
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import moviepy.editor as mp
import tempfile

# Download the YouTube video using pytube
video_url = "https://www.youtube.com/watch?v=M6VWprJ8sgI"
youtube = pytube.YouTube(video_url)
audio_stream = youtube.streams.filter(only_audio=True).first()
audio_stream.download(output_path="./", filename="audio.mp3")

# Load the wav2vec2 model and processor
model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-polish'
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Load the audio file
audio_file_path = "/Users/filip.dahlberg/gitrepos/ml-playground/transcribe-youtube/audio.mp3"  # Updated file path to reflect download location
audio = mp.AudioFileClip(audio_file_path)
with tempfile.NamedTemporaryFile(suffix='.wav') as fp:
    audio.write_audiofile(fp.name, fps=16000, nbytes=2, codec='pcm_s16le')
    waveform, sample_rate = sf.read(fp.name)

# Resample if necessary
if sample_rate != processor.feature_extractor.sampling_rate:
    resampler = sf.resample(waveform, processor.feature_extractor.sampling_rate / sample_rate)
    waveform = torch.FloatTensor(resampler)
    
input_values = processor(waveform, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors='pt').input_values
# Reshape input_values tensor to expected shape
input_values = input_values.view(1, -1)

# Transcribe the audio using the wav2vec2 model
with torch.no_grad():
    logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

# Print the transcript to the console
with open("output_file.txt", "w", encoding="utf-8") as f:
    f.write(transcription)
