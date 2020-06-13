from pydub import AudioSegment

f = AudioSegment.from_wav("1522.wav")
f = f.set_channels(1)
f.export("1522mono.wav",format = "wav")


