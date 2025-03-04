import onnx
from onnx import TensorProto

def dtype_to_string(dtype):
    mapping = {
        TensorProto.UNDEFINED: "UNDEFINED",
        TensorProto.FLOAT: "FLOAT",
        TensorProto.UINT8: "UINT8",
        TensorProto.INT8: "INT8",
        TensorProto.UINT16: "UINT16",
        TensorProto.INT16: "INT16",
        TensorProto.INT32: "INT32",
        TensorProto.INT64: "INT64",
        TensorProto.STRING: "STRING",
        TensorProto.BOOL: "BOOL",
        TensorProto.FLOAT16: "FLOAT16",
        TensorProto.DOUBLE: "DOUBLE",
        TensorProto.UINT32: "UINT32",
        TensorProto.UINT64: "UINT64",
        TensorProto.COMPLEX64: "COMPLEX64",
        TensorProto.COMPLEX128: "COMPLEX128",
        TensorProto.BFLOAT16: "BFLOAT16"
    }
    return mapping.get(dtype, "UNKNOWN")

def inspect_model_data_types(model):
    print("=== Graph Inputs ===")
    for inp in model.graph.input:
        elem_type = inp.type.tensor_type.elem_type
        print(f"Input '{inp.name}': {dtype_to_string(elem_type)}")
    
    print("\n=== Graph Outputs ===")
    for out in model.graph.output:
        elem_type = out.type.tensor_type.elem_type
        print(f"Output '{out.name}': {dtype_to_string(elem_type)}")
    
    print("\n=== Graph Value Info ===")
    for vi in model.graph.value_info:
        elem_type = vi.type.tensor_type.elem_type
        print(f"Tensor '{vi.name}': {dtype_to_string(elem_type)}")
    
    print("\n=== Initializers ===")
    for init in model.graph.initializer:
        print(f"Initializer '{init.name}': shape {list(init.dims)}, type {dtype_to_string(init.data_type)}")

# Load your ONNX model (adjust the filename as needed)
model_path = "model.onnx"
model = onnx.load(model_path)

inspect_model_data_types(model)
