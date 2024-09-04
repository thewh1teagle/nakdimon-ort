# pip3 install tensorflow==2.15.0 tf2onnx onnx
# wget https://github.com/elazarg/nakdimon/blob/master/nakdimon/Nakdimon.h5
# python main.py Nakdimon.h5
# Note: model requires special load function so I converted it using nakdimon own functions to load it.

import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from keras.models import load_model
import sys

path = sys.argv[1]
model = load_model(path)

print("model loaded")
onnx_model, _ = tf2onnx.convert.from_keras(model, ustom_objects={"loss": None})
onnx.save(onnx_model, "nakdimon.onnx")
