# pip install onnxruntime
# wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx

import onnxruntime as ort
import numpy as np
from utils import pad
import json
import time

MAX_LEN = 10000
MODEL_PATH = 'nakdimon.onnx'
CONFIG_PATH = 'config.json'
TEXT = "שלום עולם"
CONFIG: dict = None

def create_input(text: str):
    input = []
    for c in text:
        letters: list = CONFIG['letters']
        id = letters.index(c)
        input.append(id)
    input = np.array([input])
    return pad(input, MAX_LEN, dtype=np.float32)

def from_categorical(t):
    return np.argmax(t, axis=-1)        
        
def create_output(text: str, text_tensor, outputs: list) -> str:
    actual_niqqud, actual_dagesh, actual_sin = from_categorical(outputs[0]), from_categorical(outputs[1]), from_categorical(outputs[2])
    text = merge_unconditional(text, text_tensor, actual_niqqud, actual_dagesh, actual_sin)
    text = ''.join(text)
    return text

def merge_unconditional(texts, tnss, nss, dss, sss):
    global CONFIG
    res = []
    # text, niqqud, sin
    for ts, tns, ns, ds, ss in zip(texts, tnss, nss, dss, sss):
        sentence = []
        for t, tn, n, d, s in zip(ts, tns, ns, ds, ss):
            if tn == 0:
                break
            sentence.append(t)
            sentence.append(CONFIG['dagesh'][d] if t in  CONFIG['rules']['can_dagesh'] else '\uFEFF')
            sentence.append(CONFIG['sin'][d] if t in  CONFIG['rules']['can_niqqud'] else '\uFEFF')
            sentence.append(CONFIG['niqqud'][d] if t in  CONFIG['rules']['can_sin'] else '\uFEFF')
        res.append(''.join(sentence))
    return res

def read_config():
    global CONFIG
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)

if __name__ == '__main__':
    read_config()
    input = create_input(TEXT)
    input = input.reshape(1, -1)
    start_t = time.time()
    session = ort.InferenceSession(MODEL_PATH)
    outputs = session.run(None, {'input_1': input})
    text = create_output(pad(list(TEXT), dtype='<U1', value=0, maxlen=MAX_LEN), input, outputs)
    print('Took {}', time.time() - start_t)
    print(text)