
import whisper
# Python code to convert video to audio
import moviepy.editor as mp

# Insert Local Video File Path
clip = mp.VideoFileClip("//Users/dipanshukumar/Desktop/RED_BOT/Deep_learning.mp4")

# Insert Local Audio File Path
clip.audio.write_audiofile("//Users/dipanshukumar/Desktop/RED_BOT/Deep_learning.wav",codec='pcm_s16le')



#Change speech to text

model = whisper.load_model('small')

text = model.transcribe('//Users/dipanshukumar/Desktop/RED_BOT/Deep_learning.wav')

with open('//Users/dipanshukumar/Desktop/RED_BOT/transcribe1.txt','w+') as file:
    file.write(text['text'])