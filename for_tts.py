from gtts import gTTS
text ="놀람"

tts = gTTS(text=text, lang='ko')
tts.save("./tts/surprise.mp3")