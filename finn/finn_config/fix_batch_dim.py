import onnx

# Load your ONNX model (adjust the path/filename as needed)
model = onnx.load('/home/eveneiha/finn/finn_config/model.onnx')

# Identify the model input(s) that require a fixed batch dimension.
# Typically, your primary input is named "quant_input" (or similar).
for inp in model.graph.input:
    if inp.name == "quant_input":
        # Print the original shape for debugging.
        original_dims = [d.dim_value if d.HasField("dim_value") else "unk" for d in inp.type.tensor_type.shape.dim]
        print("Original input shape for {}: {}".format(inp.name, original_dims))
        
        # Set the batch dimension (first dimension) to 1.
        if len(inp.type.tensor_type.shape.dim) > 0:
            inp.type.tensor_type.shape.dim[0].dim_value = 1
        else:
            print("Warning: No dimensions found for input {}".format(inp.name))
        # Print the updated shape.
        updated_dims = [d.dim_value if d.HasField("dim_value") else "unk" for d in inp.type.tensor_type.shape.dim]
        print("Updated input shape for {}: {}".format(inp.name, updated_dims))
        break  # Assuming only one primary input needs adjustment.

# Save the modified model
onnx.save(model, '/home/eveneiha/finn/finn_config/model_fixed_batch.onnx')
print("Saved fixed model as model_fixed_batch.onnx")
