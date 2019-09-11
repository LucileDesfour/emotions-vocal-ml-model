#!/usr/bin/python

from flask import Flask
from flask import request
from vocalEmotionRecognition import analyse_emotions
app = Flask(__name__)

@app.route('/analyse/emotions/vocal', methods=['POST'])
def hello_world():
  try:
    file = request.files['wav']
  except:
    file = None
  if file == None:
    return 'expected wav file in body'
  print(file)
  filename = file.filename
  file.save('./Audio_Speech_Actors/training/' + filename)
  #analyseEmotions.analyse_emotions(wav)
  print('/Audio_Speech_Actors/training/' + filename)
  return (analyse_emotions('./Audio_Speech_Actors/training/' + filename))
