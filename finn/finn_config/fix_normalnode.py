import onnx
from onnx import helper, shape_inference

def update_random_normal_shape(model, old_shape, new_shape):
    """
    For every RandomNormal node that has an attribute "shape" matching old_shape,
    update it to new_shape.
    Both old_shape and new_shape should be lists of integers.
    """
    for node in model.graph.node:
        if node.op_type == "RandomNormal":
            for attr in node.attribute:
                if attr.name == "shape":
                    current_shape = list(attr.ints)
                    if current_shape == old_shape:
                        print(f"Updating RandomNormal node {node.name or '<anonymous>'} shape from {current_shape} to {new_shape}")
                        attr.ints[:] = new_shape
    return model

# Load your ONNX model.
model_path = "model.onnx"  # or whichever file you produced after converting convs.
model = onnx.load(model_path)

# Suppose the problematic RandomNormal has shape [1, 256] but should be [1, 260].
old_shape = [1, 256]
new_shape = [1, 260]

model = update_random_normal_shape(model, old_shape, new_shape)

# Run shape inference to update metadata.
model = shape_inference.infer_shapes(model)
# Save the updated model.
new_model_path = "model_conv2d_fixed.onnx"
onnx.save(model, new_model_path)
print("Saved updated model as", new_model_path)
