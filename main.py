import json
import numpy as np
import onnxruntime as ort
import time

class Nakdimon:
    def __init__(self, model_path, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.RAFE = self.config['RAFE']
        self.niqqud = self.config['niqqud']
        self.dagesh = self.config['dagesh']
        self.sin = self.config['sin']
        self.HEBREW_LETTERS = self.config['HEBREW']
        self.VALID_LETTERS = self.config['VALID'] + self.HEBREW_LETTERS
        self.SPECIAL_TOKENS = self.config['SPECIAL']
        self.NORMALIZE_MAP = self.config['normalize_map']
        self.NORMALIZE_DEFAULT_VALUE = self.NORMALIZE_MAP['DEFAULT']
        self.CAN_DAGESH = self.config['can_dagesh']
        self.CAN_SIN = self.config['can_sin']
        self.CAN_NIQQUD = self.config['can_niqqud']
        self.ALL_TOKENS = [''] + self.SPECIAL_TOKENS + self.VALID_LETTERS
        self.MAXLEN = self.config['MAXLEN']
        self.session = ort.InferenceSession(model_path)

    def normalize(self, c):
        if c in self.VALID_LETTERS:
            return c
        return self.NORMALIZE_MAP.get(c, self.NORMALIZE_DEFAULT_VALUE)

    def split_to_rows(self, text):
        space = self.ALL_TOKENS.index(" ")
        arr = [[self.ALL_TOKENS.index(c) for c in s] for s in text.split(" ")]
        rows = []
        line = []
        for tokens in arr:
            if len(tokens) + len(line) + 1 > self.MAXLEN:
                while len(line) < self.MAXLEN:
                    line.append(0)
                rows.append(line)
                line = []
            line.extend(tokens + [space])
        while len(line) < self.MAXLEN:
            line.append(0)
        rows.append(line)
        return np.array(rows)

    def from_categorical(self, arr):
        return np.argmax(arr, axis=-1).flatten()

    def prediction_to_text(self, prediction, undotted_text):
        niqqud, dagesh, sin = prediction
        niqqud_result = self.from_categorical(niqqud)
        dagesh_result = self.from_categorical(dagesh)
        sin_result = self.from_categorical(sin)

        output = []
        for i, c in enumerate(undotted_text):
            fresh = {'char': c, 'niqqud': '', 'dagesh': '', 'sin': ''}
            if c in self.HEBREW_LETTERS:
                if c in self.CAN_NIQQUD:
                    fresh['niqqud'] = self.niqqud[niqqud_result[i]]
                if c in self.CAN_DAGESH:
                    fresh['dagesh'] = self.dagesh[dagesh_result[i]]
                if c in self.CAN_SIN:
                    fresh['sin'] = self.sin[sin_result[i]]
            output.append(fresh)
        return output

    def remove_niqqud(self, text):
        return ''.join([c for c in text if c not in '\u0591-\u05C7'])

    def to_text(self, item):
        c = '\r\n' if item['char'] == '\n' else item['char']
        return c + (item['dagesh'] or '') + (item['sin'] or '') + (item['niqqud'] or '')

    def update_dotted(self, items):
        return ''.join([self.to_text(item) for item in items])

    def compute(self, text):
        undotted_text = self.remove_niqqud(text)
        input_data = self.split_to_rows(''.join(map(self.normalize, undotted_text)))
        input_tensor = np.array(input_data, dtype=np.float32)
        prediction = self.session.run(None, {"input_1": input_tensor})
        res = self.prediction_to_text(prediction, undotted_text)
        return self.update_dotted(res)


if __name__ == '__main__':
    nakdimon = Nakdimon("nakdimon.onnx", "config.json")
    text = 'שלום עולם!'
    start_t = time.time()
    dotted_text = nakdimon.compute(text)
    print(dotted_text)
    print(f'Took {time.time() - start_t:.1} seconds.')
