from huggingsound import SpeechRecognitionModel
import os
import datetime

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-polish")
audio_paths = ["/Users/filip.dahlberg/gitrepos/ml-playground/split-audio/audio.m4a"]

# Set the path to the directory you want to list
folder_path = "/Users/filip.dahlberg/gitrepos/ml-playground/split-audio/output_audio"

# Create a dictionary to store file creation dates
creation_dates = {}

# Use the listdir() function to get a list of files in the directory
files = os.listdir(folder_path)

# Iterate through files and get creation dates
for file in files:
    file_path = os.path.join(folder_path, file)
    creation_time = os.path.getctime(file_path)
    creation_dates[file] = datetime.datetime.fromtimestamp(creation_time)

# Sort files by creation date
sorted_files = sorted(creation_dates, key=creation_dates.get)

# Print sorted filenames
for file in sorted_files:
    print(file)

# i=0
for file in sorted_files:
    # i+=1
    transcriptions = model.transcribe([os.path.join(folder_path, file)])
    transcriptions[0]["transcription"] = transcriptions[0]["transcription"] + " "
    print(file)
    with open("output_file.txt", "a", encoding="utf-8") as f:
        f.write(transcriptions[0]["transcription"])
    # if i == 1:
        # break
    

# Combine the transcriptions
print(transcriptions)

# Print the transcript to the console
