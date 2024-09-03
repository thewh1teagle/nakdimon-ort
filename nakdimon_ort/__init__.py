import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

class Nakdimon:
    def __init__(self, model_path, config_path):
        self.config = self.load_config(model_path, config_path)
        
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
        self.REMOVE_NIQQUD_RANGE = self.config['remove_niqqud_range']
        self.MAXLEN = self.config['MAXLEN']
        self.session = ort.InferenceSession(model_path)

    def load_config(self, model_path, config_path):
        if not Path(config_path).exists() and Path('assets/config.json').exists():
            # Just make development a bit better
            config_path = 'assets/config.json'

        assert Path(model_path).exists(), (
            f"Configuration file not found: {config_path}\n"
            "Please download the Nakdimon model before executing.\n"
            "You can download it using the following command:\n"
            "wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx"
        )

        assert Path(config_path).exists(), (
            f"Configuration file not found: {config_path}\n"
            "Please download the Nakdimon configuration file before executing.\n"
            "You can download it using the following command:\n"
            "wget https://github.com/thewh1teagle/nakdimon-ort/raw/main/assets/config.json"
        )

        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def normalize(self, c):
        return self.NORMALIZE_MAP.get(c, self.NORMALIZE_DEFAULT_VALUE) if c not in self.VALID_LETTERS else c

    def split_to_rows(self, text):
        space_id = self.ALL_TOKENS.index(" ")  # Index of the space character in tokens
        word_ids_matrix = [[self.ALL_TOKENS.index(c) for c in word] for word in text.split()]  # Convert text to token IDs
        
        rows, cur_row = [], []

        for word_ids in word_ids_matrix:
            # Check if adding the word exceeds the max length
            if len(cur_row) + len(word_ids) + 1 > self.MAXLEN:
                rows.append(cur_row + [0] * (self.MAXLEN - len(cur_row)))  # Pad and save the current row
                cur_row = []
            cur_row.extend(word_ids + [space_id])  # Add the word and space to the current row

        # Final padding and appending of the last row
        rows.append(cur_row + [0] * (self.MAXLEN - len(cur_row)))
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
        return ''.join([c for c in text if not (self.REMOVE_NIQQUD_RANGE[0] <= c <= self.REMOVE_NIQQUD_RANGE[1])])

    def to_text(self, item):
        return item['char'] + (item['dagesh'] or '') + (item['sin'] or '') + (item['niqqud'] or '')

    def update_dotted(self, items):
        return ''.join([self.to_text(item) for item in items])

    def compute(self, text):
        undotted = self.remove_niqqud(text)
        normalized = ''.join(map(self.normalize, undotted))
        input_matrix = self.split_to_rows(normalized)
        input_tensor = np.array(input_matrix, dtype=np.float32)
        prediction = self.session.run(None, {"input_1": input_tensor})
        res = self.prediction_to_text(prediction, undotted)
        return self.update_dotted(res)


