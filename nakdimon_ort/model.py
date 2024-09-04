import numpy as np
import onnxruntime as ort
from .config import load_config, DEFAULT_CONFIG_PATH


class Nakdimon:
    def __init__(self, model_path, config_path=DEFAULT_CONFIG_PATH):
        self.config = load_config(model_path, config_path)
        self.rafe = self.config["rafe"]
        self.niqqud = self.config["niqqud"]
        self.dagesh = self.config["dagesh"]
        self.sin = self.config["sin"]
        self.hebrew_letters = self.config["hebrew"]
        self.valid = self.config["valid"] + self.hebrew_letters
        self.special = self.config["special"]
        self.normalize_map = self.config["normalize_map"]
        self.normalize_deafult_value = self.normalize_map["default"]
        self.can_dagesh = self.config["can_dagesh"]
        self.can_sin = self.config["can_sin"]
        self.can_niqqud = self.config["can_niqqud"]
        self.all_tokens = [""] + self.special + self.valid
        self.remove_niqqud_range = self.config["remove_niqqud_range"]
        self.max_len = self.config["max_len"]
        self.session = ort.InferenceSession(model_path)

    def normalize(self, c):
        return (
            self.normalize_map.get(c, self.normalize_deafult_value)
            if c not in self.valid
            else c
        )

    def split_to_rows(self, text):
        word_ids_rows = [
            [self.all_tokens.index(c) for c in word] for word in text.split(" ")
        ]  # Convert text to token IDs
        rows, cur_row = [], []

        for word_ids in word_ids_rows:
            # Check if adding the word exceeds the max length
            if len(cur_row) + len(word_ids) + 1 > self.max_len:
                padding = [0] * (self.max_len - len(cur_row))
                rows.append(cur_row + padding)  # Pad and save the current row
                cur_row = []
            cur_row.extend(
                word_ids + [self.all_tokens.index(" ")]
            )  # Add the word and space to the current row

        # Final padding and appending of the last row
        rows.append(cur_row + [0] * (self.max_len - len(cur_row)))
        return rows

    def from_categorical(self, arr):
        return np.argmax(arr, axis=-1).flatten()

    def prediction_to_text(self, input, prediction, undotted_text):
        niqqud, dagesh, sin = prediction
        niqqud_result = self.from_categorical(niqqud)
        dagesh_result = self.from_categorical(dagesh)
        sin_result = self.from_categorical(sin)
        output = []
        for i, c in enumerate(undotted_text):
            fresh = {"char": c, "niqqud": "", "dagesh": "", "sin": ""}
            if c in self.hebrew_letters:
                if c in self.can_niqqud:
                    fresh["niqqud"] = self.niqqud[niqqud_result[i]]
                if c in self.can_dagesh:
                    fresh["dagesh"] = self.dagesh[dagesh_result[i]]
                if c in self.can_sin:
                    fresh["sin"] = self.sin[sin_result[i]]
            output.append(fresh)
        return output

    def remove_niqqud(self, text):
        return "".join(
            [
                c
                for c in text
                if not (self.remove_niqqud_range[0] <= c <= self.remove_niqqud_range[1])
            ]
        )

    def to_text(self, item):
        return (
            item["char"]
            + (item["dagesh"] or "")
            + (item["sin"] or "")
            + (item["niqqud"] or "")
        )

    def update_dotted(self, items):
        return "".join([self.to_text(item) for item in items])

    def compute(self, text):
        undotted = self.remove_niqqud(text)
        normalized = "".join(map(self.normalize, undotted))
        input = self.split_to_rows(normalized)
        input_tensor = np.array(input, dtype=np.float32)
        prediction = self.session.run(None, {"input_1": input_tensor})
        res = self.prediction_to_text(input, prediction, undotted)
        return self.update_dotted(res)
