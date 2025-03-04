import onnx

def get_shape_str(dim_list):
    """Helper to convert dimension list to a readable string."""
    dims = []
    for d in dim_list:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"

def print_graph_shapes(model, title="Graph Shapes"):
    print(f"\n==== {title} ====")
    # Print inputs
    print("\nGraph Inputs:")
    for inp in model.graph.input:
        shape = get_shape_str(inp.type.tensor_type.shape.dim)
        print(f"  {inp.name}: {shape}")
    # Print value_info
    print("\nValue Info:")
    for vi in model.graph.value_info:
        shape = get_shape_str(vi.type.tensor_type.shape.dim)
        print(f"  {vi.name}: {shape}")
    # Print outputs
    print("\nGraph Outputs:")
    for out in model.graph.output:
        shape = get_shape_str(out.type.tensor_type.shape.dim)
        print(f"  {out.name}: {shape}")

# Path to your model file; adjust if necessary.
model_path = "/home/eveneiha/finn/finn_config/model.onnx"

# Load the existing model.
model_existing = onnx.load(model_path)
print_graph_shapes(model_existing, "Existing Shapes (Before Inference)")

# Run ONNX shape inference.
model_inferred = onnx.shape_inference.infer_shapes(model_existing)
print_graph_shapes(model_inferred, "Inferred Shapes (After Shape Inference)")
