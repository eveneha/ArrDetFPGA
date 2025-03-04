import onnx
from onnx import helper, TensorProto

def insert_streaming_dataflow_partition(model):
    # Create a new StreamingDataflowPartition node.
    sdp_node = helper.make_node(
        "StreamingDataflowPartition",
        inputs=[out.name for out in model.graph.output],  # using current outputs as inputs
        outputs=["dataflow_partition_output"],
        name="StreamingDataflowPartition_0"
    )
    # Set the domain for the custom op.
    sdp_node.domain = "finn.custom_op"  # Make sure this matches the domain expected by FINN
    
    # Append the SDP node to the graph.
    model.graph.node.append(sdp_node)
    
    # Create a new graph output for the SDP node.
    new_output = helper.make_tensor_value_info("dataflow_partition_output", TensorProto.FLOAT, None)
    
    # Clear existing outputs and extend with new output.
    del model.graph.output[:]
    model.graph.output.extend([new_output])
    
    return model

# Load your model (replace with your filename)
model_path = "model.onnx"
model = onnx.load(model_path)

# Insert the StreamingDataflowPartition node with a proper domain.
model = insert_streaming_dataflow_partition(model)

# Run shape inference (optional, to update metadata)
# model = onnx.shape_inference.infer_shapes(model)

# Save the updated model.
onnx.save(model, "model_with_sdp.onnx")
print("Saved updated model as model_with_sdp.onnx")
