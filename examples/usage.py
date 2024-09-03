"""
Please download nakdimon model before execute

wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx
wget https://github.com/thewh1teagle/nakdimon-ort/raw/main/assets/config.json
python usage.py
"""

from nakdimon_ort import Nakdimon

nakdimon = Nakdimon("nakdimon.onnx", "config.json")
text = "שלום עולם!"
dotted_text = nakdimon.compute(text)
print(dotted_text)
