import whisper

model = whisper.load_model("medium")
result = model.transcribe("audio.m4a")
print(result["text"])
