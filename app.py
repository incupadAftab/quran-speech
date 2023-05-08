from werkzeug.utils import secure_filename
import librosa
from os import path
import torchaudio
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from lang_trans.arabic import buckwalter
from nltk import edit_distance
from Levenshtein import distance
from tqdm import tqdm
import pyquran as q
import numpy as np
from bidi.algorithm import get_display
import arabic_reshaper
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['Acess-Control-Allow-Origin'] = '*'

uploaded_path = os.path.join(
    app.instance_path, 
    'static'
)

try: 
    os.makedirs(uploaded_path)
except: 
    pass 



def audioToNumpyArray(path):
    # Check the number of channels in the audio
    y, sr = librosa.load(path, sr=16000, mono=False)
    # Convert stereo to mono by averaging the channels
    if y.ndim > 1:
        y = librosa.to_mono(y)

    audio_array = np.array(y, dtype=float)
    return audio_array


def load_Quran_fine_tuned_elgeish_xlsr_53_model_and_processor():
    global loaded_model, loaded_processor
    loaded_model = Wav2Vec2ForCTC.from_pretrained("Nuwaisir/Quran_speech_recognizer").eval()
    loaded_processor = Wav2Vec2Processor.from_pretrained("Nuwaisir/Quran_speech_recognizer")

def load_elgeish_xlsr_53_model_and_processor():
    global loaded_model, loaded_processor
    loaded_model = Wav2Vec2ForCTC.from_pretrained("elgeish/wav2vec2-large-xlsr-53-arabic").eval()
    loaded_processor = Wav2Vec2Processor.from_pretrained("elgeish/wav2vec2-large-xlsr-53-arabic")


def predict(single):
    inputs = loaded_processor(single["speech"], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        predicted = torch.argmax(loaded_model(inputs.input_values).logits, dim=-1)
    predicted[predicted == -100] = loaded_processor.tokenizer.pad_token_id  # see fine-tuning script
    pred_1 = loaded_processor.tokenizer.batch_decode(predicted)[0]
    print(pred_1)
    single["predicted"] = buckwalter.untrans(pred_1)
    return single


def last_para_str(taskeel=False):
    quran_string = ''
    for i in range (1, 115):
        if i == 1:
            quran_string += ' '.join(q.quran.get_sura(i, with_tashkeel=taskeel,basmalah=True))
        else :
            quran_string += ' '.join(q.quran.get_sura(i, with_tashkeel=taskeel,basmalah=False))
        
        quran_string += ' '
    return quran_string

def find_match_2(q_str, s, spaces, threshhold = 10):
  len_q = len(q_str)
  len_s = len(s)
  min_dist = 1000000000
  min_dist_pos = []
  for i in tqdm(spaces):
    j = i+1
    k = j + len_s + len_s // 3
    if k > len_q:
      break
    dist = edit_distance(q_str[j:k],s)
    if dist < min_dist:
      min_dist = dist
      min_dist_pos = [j]
    elif dist == min_dist:
      min_dist_pos.append(j)
  return min_dist, min_dist_pos

def find_all_index(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

# last_para = q.quran.get_sura(1, with_tashkeel=True,basmalah=True)
# last_para_spaces = find_all_index(last_para,' ')
# print('Last para space index = ' + str(last_para_spaces));
# last_para_spaces.insert(0, -1)

def find_the_verse(sentence, surah_number):
    last_para = q.quran.get_sura(surah_number, with_tashkeel=True,basmalah=False)
    distances = np.array([distance(sentence, s) for s in last_para])
    print(f"distances {distances}")
    most_similar_verse_number = np.argmin(distances)
    print('found index ' + str(most_similar_verse_number));
    return last_para[most_similar_verse_number], most_similar_verse_number + 1, distances[most_similar_verse_number]

def match_the_verse(sentence, surah_number, verse_number):
    verse_text = q.quran.get_verse(surah_number, verse_number, with_tashkeel=True)
    dist = distance(sentence, verse_text)
    return verse_text, dist

@app.route("/recognize/", methods=["POST"])
def pipeline():
    audio_file = request.files.get("audio")
    sura_no = request.form['sura']
    verse_no = request.form['verse_no']
    #sura_no = request.json["sura"]
    if audio_file and audio_file.filename != '': 
        audio_path = os.path.join(
            uploaded_path,
            secure_filename(audio_file.filename)
        )
        # Save the file on the server.
        audio_file.save(audio_path)
    
    verse_recording = audioToNumpyArray(audio_path)
    single_example = {
        "speech": verse_recording,
        "sampling_rate": 16000,
    }
    predicted = predict(single_example)
    reshaped_text = arabic_reshaper.reshape(predicted['predicted'])
    bidi_text = get_display(reshaped_text)
    verse, verse_number, distances = find_the_verse(predicted['predicted'], int(sura_no))
    # dist, poses = find_match_2(last_para, predicted['predicted'], spaces=last_para_spaces)
    # print("distance:",dist)
    # print("number of matches:", len(poses))
    # print('poses array ' + str(poses))

    response = {
        "transcription": predicted["predicted"],
        "verse":str(verse),
        "number":str(verse_number),
        "distance": str(distances),
        "bidi_text":str(bidi_text)
    }

    return jsonify(response)
#    wrong_word = find_the_wrong_word((predicted['predicted'], verse))

    #dist, poses = find_match_2(last_para, predicted['predicted'], spaces=last_para_spaces)
    #print("distance:",dist)
    #print("number of matches:", len(poses))
    #print('poses array ' + str(poses))
    #for i in poses:
    #    print(last_para[i],'\n')


"""
### Load the elgeish_xlsr_53 model
"""

load_Quran_fine_tuned_elgeish_xlsr_53_model_and_processor()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000", debug=True)

