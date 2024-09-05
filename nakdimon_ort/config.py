import json
from pathlib import Path

class Config:
    def __init__(self, model_path, config_path: str) -> None:
        config = self.load(model_path, config_path)
        self.rafe: str = config["rafe"]
        self.niqqud: list = config["niqqud"]
        self.dagesh: list = config["dagesh"]
        self.sin: list = config["sin"]
        self.hebrew_letters: list = config["hebrew"]
        self.valid: list = config["valid"] + self.hebrew_letters
        self.special: list = config["special"]
        self.normalize_map: dict = config["normalize_map"]
        self.normalize_deafult_value: str = self.normalize_map["default"]
        self.can_dagesh: list = config["can_dagesh"]
        self.can_sin: list = config["can_sin"]
        self.can_niqqud: list = config["can_niqqud"]
        self.all_tokens: list = [""] + self.special + self.valid
        self.remove_niqqud_range: list = config["remove_niqqud_range"]
        self.max_len: int = config["max_len"]
        
    def load(self, model_path, config_path):
        assert Path(model_path).exists(), (
            f"Model file not found: {model_path}\n"
            "Please download the Nakdimon model before executing.\n"
            "You can download it using the following command:\n"
            "wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx"
        )

        assert Path(config_path).exists(), (
            f"Configuration file not found: {config_path}\n"
            "Please download the Nakdimon configuration file before executing.\n"
            "You can download it using the following command:\n"
            "wget https://github.com/thewh1teagle/nakdimon-ort/raw/main/nakdimon_ort/config.json"
        )

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
