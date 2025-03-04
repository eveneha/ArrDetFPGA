#!/usr/bin/env python3
import onnx
from qonnx.core.modelwrapper import ModelWrapper

# Change the filename to match your model file.
model_filename = "/home/eveneiha/finn/finn_config/model.onnx"

# Load the model using ModelWrapper, which provides a handy method to get tensor shapes.
model = ModelWrapper(model_filename)

print("=== Model Node Shapes ===")
for node in model.graph.node:
    print(f"Node: {node.name} (Op: {node.op_type})")
    # Print shapes for each input of the node.
    for inp in node.input:
        try:
            shape = model.get_tensor_shape(inp)
        except Exception as e:
            shape = f"ERROR: {e}"
        print(f"  Input: {inp} -> Shape: {shape}")
    # Print shapes for each output of the node.
    for out in node.output:
        try:
            shape = model.get_tensor_shape(out)
        except Exception as e:
            shape = f"ERROR: {e}"
        print(f"  Output: {out} -> Shape: {shape}")
    print("-" * 40)

print("=== End of Model ===")
