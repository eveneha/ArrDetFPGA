import onnx

model_path = "model_v14.onnx"  # Your model after converting Conv1d to Conv2d
model = onnx.load(model_path)

for node in model.graph.node:
    for idx, inp in enumerate(node.input):
        print(f"Node '{node.name}' (op_type={node.op_type}), input index {idx}: {inp}")
