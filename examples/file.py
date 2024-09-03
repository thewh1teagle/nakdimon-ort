"""
Please download nakdimon model before execute

wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx
wget https://github.com/thewh1teagle/nakdimon-ort/raw/main/assets/config.json
python usage.py input.txt output.txt
"""

from nakdimon_ort import Nakdimon
import sys

input = sys.argv[1]
output = sys.argv[2]

nakdimon = Nakdimon("nakdimon.onnx", "config.json")
text = open(input, "r", encoding="utf-8").read()
dotted_text = nakdimon.compute(text)
with open(output, "w", encoding="utf-8") as f:
    f.write(dotted_text)
