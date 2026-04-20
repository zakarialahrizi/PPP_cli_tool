"""
Run this script once to generate a dummy model.onnx for testing:

    pip install onnx numpy
    python generate_dummy_model.py

It produces malvis/model.onnx — a tiny linear layer that accepts
the same (1, 1, 224, 224) float32 input that inference.py sends,
and returns a (1, 5) logits tensor. No GPU or training required.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

NUM_CLASSES = 5   # must match len(CLASSES) in inference.py
IMAGE_SIZE  = 224

# Flatten size: 1 * 224 * 224
flat = IMAGE_SIZE * IMAGE_SIZE

# Weight matrix (flat → NUM_CLASSES) — random but fixed for reproducibility
rng = np.random.default_rng(42)
W = rng.standard_normal((flat, NUM_CLASSES)).astype(np.float32)
b = np.zeros(NUM_CLASSES, dtype=np.float32)

# ONNX initializers
W_init = numpy_helper.from_array(W, name="W")
b_init = numpy_helper.from_array(b, name="b")

# Graph nodes
flatten_node = helper.make_node(
    "Flatten",
    inputs=["image"],
    outputs=["flat"],
    axis=1,          # flatten all dims after batch
)
gemm_node = helper.make_node(
    "Gemm",
    inputs=["flat", "W", "b"],
    outputs=["logits"],
    transB=0,
)

# Graph I/O
image_input = helper.make_tensor_value_info(
    "image", TensorProto.FLOAT, [1, 1, IMAGE_SIZE, IMAGE_SIZE]
)
logits_output = helper.make_tensor_value_info(
    "logits", TensorProto.FLOAT, [1, NUM_CLASSES]
)

graph = helper.make_graph(
    [flatten_node, gemm_node],
    "dummy_malware_cnn",
    [image_input],
    [logits_output],
    initializer=[W_init, b_init],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)

out = "malvis/model.onnx"
onnx.save(model, out)
print(f"Saved dummy model → {out}")
print(f"  Input : image  (1, 1, {IMAGE_SIZE}, {IMAGE_SIZE})  float32")
print(f"  Output: logits (1, {NUM_CLASSES})                    float32")
