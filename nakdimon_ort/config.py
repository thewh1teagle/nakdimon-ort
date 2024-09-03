import json
from pathlib import Path


def load_config(model_path, config_path):
    if not Path(config_path).exists() and Path("assets/config.json").exists():
        # Just make development a bit better
        config_path = "assets/config.json"

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

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
