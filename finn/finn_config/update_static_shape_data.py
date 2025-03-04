import onnx

def update_tensor_shape(model, tensor_name, new_shape):
    updated = False
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            dims = vi.type.tensor_type.shape.dim
            del dims[:]
            for dim in new_shape:
                d = onnx.TensorShapeProto.Dimension()
                d.dim_value = dim
                dims.append(d)
            print(f"Updated value_info for {tensor_name} to {new_shape}")
            updated = True
    return model, updated

model_path = "model.onnx"
model = onnx.load(model_path)

# For example, update the Squeeze output which should be [1, 16]:
model, updated = update_tensor_shape(model, "/Squeeze_output_0", [1, 16])
if not updated:
    print("Warning: /Squeeze_output_0 not found in value_info.")

# Run shape inference to propagate the new static shapes.
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "model_v3_static.onnx")
print("Saved updated model as model_v3_static.onnx")
