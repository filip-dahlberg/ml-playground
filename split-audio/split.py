from pydub import AudioSegment
import os

# specify the input file path and duration of each split in milliseconds
input_file = "audio.m4a"
split_duration = 200000

# create an AudioSegment object from the input file
audio = AudioSegment.from_file(input_file, format="m4a")

# get the total duration of the audio file in milliseconds
total_duration = len(audio)

# create a directory to store the output files
os.makedirs("output_audio", exist_ok=True)

# split the audio file into multiple segments and save them as separate files
for i in range(0, total_duration, split_duration):
    # calculate the start and end time of the segment
    start_time = i
    end_time = min(i + split_duration, total_duration)

    # extract the segment and save it as a separate file
    output_file = f"output_audio/{start_time}-{end_time}.mp4"
    audio_segment = audio[start_time:end_time]
    audio_segment.export(output_file, format="mp4")
