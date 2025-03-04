import onnx
from onnx import helper, shape_inference

def convert_conv1d_to_conv2d(model):
    # Update each Conv node's attributes.
    for node in model.graph.node:
        if node.op_type == "Conv":
            for attr in node.attribute:
                if attr.name == "kernel_shape" and len(attr.ints) == 1:
                    k = attr.ints[0]
                    attr.ints[:] = [1, k]
                    print(f"Updated kernel_shape for node {node.name} to [1, {k}]")
                if attr.name == "pads" and len(attr.ints) == 2:
                    p = attr.ints[0]
                    attr.ints[:] = [0, p, 0, p]
                    print(f"Updated pads for node {node.name} to [0, {p}, 0, {p}]")
                if attr.name == "strides" and len(attr.ints) == 1:
                    s = attr.ints[0]
                    attr.ints[:] = [1, s]
                    print(f"Updated strides for node {node.name} to [1, {s}]")
    # Update initializers for conv weights.
    for init in model.graph.initializer:
        if len(init.dims) == 3:
            dims = list(init.dims)  # [out_channels, in_channels, kernel_size]
            dims.insert(2, 1)        # becomes [out_channels, in_channels, 1, kernel_size]
            init.dims[:] = dims
            print(f"Updated initializer {init.name} shape to {dims}")
    return model

# Load your exported model.
model = onnx.load("model.onnx")
# Convert all 1D convs to 2D representations.
model = convert_conv1d_to_conv2d(model)
# Run shape inference to update all metadata.
model = shape_inference.infer_shapes(model)
# Save the updated model.
onnx.save(model, "model_conv2d.onnx")
print("Saved updated model as model_conv2d.onnx")
