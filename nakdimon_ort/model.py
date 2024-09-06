import numpy as np
import onnxruntime as ort
from .config import Config
from pathlib import Path


class Nakdimon:
    def __init__(self, model_path, config_path=Path(__file__).parent / "config.json"):
        self.config = Config(model_path, config_path)
        self.session = ort.InferenceSession(model_path)

    def normalize(self, c):
        return (
            self.config.normalize_map.get(c, self.config.normalize_deafult_value)
            if c not in self.config.valid
            else c
        )

    def split_to_rows(self, text):
        word_ids_rows = [
            [self.config.all_tokens.index(c) for c in word] for word in text.split(" ")
        ]  # Convert text to token IDs
        rows, cur_row = [], []

        for word_ids in word_ids_rows:
            # Check if adding the word exceeds the max length
            if len(cur_row) + len(word_ids) + 1 > self.config.max_len:
                padding = [0] * (self.config.max_len - len(cur_row))
                rows.append(cur_row + padding)  # Pad and save the current row
                cur_row = []
            cur_row.extend(
                word_ids + [self.config.all_tokens.index(" ")]
            )  # Add the word and space to the current row

        # Final padding and appending of the last row
        rows.append(cur_row + [0] * (self.config.max_len - len(cur_row)))
        return rows

    def from_categorical(self, input_tensor, arr: list):
        # Filter zeros from input_tensor
        return np.argmax(arr[input_tensor > 0], axis=-1).flatten()

    def prediction_to_text(self, input_tensor, prediction, undotted_text):
        niqqud, dagesh, sin = prediction
        niqqud_result = self.from_categorical(input_tensor, niqqud)
        dagesh_result = self.from_categorical(input_tensor, dagesh)
        sin_result = self.from_categorical(input_tensor, sin)
        output = []
        for i, c in enumerate(undotted_text):
            fresh = {"char": c, "niqqud": "", "dagesh": "", "sin": ""}
            if c in self.config.hebrew_letters:
                if c in self.config.can_niqqud:
                    fresh["niqqud"] = self.config.niqqud[niqqud_result[i]]
                if c in self.config.can_dagesh:
                    fresh["dagesh"] = self.config.dagesh[dagesh_result[i]]
                if c in self.config.can_sin:
                    fresh["sin"] = self.config.sin[sin_result[i]]
            output.append(fresh)
        return output

    def remove_niqqud(self, text):
        return "".join(
            [
                c
                for c in text
                if not (
                    self.config.remove_niqqud_range[0]
                    <= c
                    <= self.config.remove_niqqud_range[1]
                )
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
        res = self.prediction_to_text(input_tensor, prediction, undotted)
        return self.update_dotted(res)
