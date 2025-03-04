import onnx
from onnx import helper, TensorProto
from qonnx.transformation.base import Transformation

class InsertStreamingDataflowPartition(Transformation):
    def apply(self, model):
        # Check that the model has a single input.
        if len(model.graph.input) != 1:
            raise ValueError("InsertStreamingDataflowPartition supports models with a single input.")
        orig_input = model.graph.input[0].name
        print(f"Original input name: {orig_input}")

        # Define the new output name for the SDP node.
        sdp_output = "sdp_output"

        # Create a new node. We start with an Identity node and then change its op_type.
        sdp_node = helper.make_node(
            "Identity",
            inputs=[orig_input],
            outputs=[sdp_output],
            name="StreamingDataflowPartition_Node"
        )
        # Mark it as a streaming dataflow partition.
        sdp_node.op_type = "StreamingDataflowPartition"
        print("Created SDP node with output:", sdp_output)

        # Insert the SDP node at the beginning of the graph.
        model.graph.node.insert(0, sdp_node)

        # Rewire all nodes that refer to the original input.
        # Often, node inputs might be "quant_input" or "quant_input:0", so we use startswith().
        for node in model.graph.node:
            new_inputs = []
            for inp in node.input:
                if inp.startswith(orig_input):
                    new_inputs.append(sdp_output)
                else:
                    new_inputs.append(inp)
            node.input[:] = new_inputs

        # Add a value_info for the SDP node's output.
        # Adjust the shape if your converted model expects a different shape.
        # Here we assume the input to your Conv2d model is [1, 1, 1, 256].
        new_vi = helper.make_tensor_value_info(sdp_output, TensorProto.FLOAT, [1, 1, 1, 256])
        model.graph.value_info.append(new_vi)
        print(f"Added value_info for {sdp_output} with shape [1, 1, 1, 256]")

        # Replace the model's outputs with the SDP node's output.
        del model.graph.output[:]
        new_output_vi = helper.make_tensor_value_info(sdp_output, TensorProto.FLOAT, [1, 1, 1, 256])
        model.graph.output.extend([new_output_vi])
        print(f"Replaced model output with {sdp_output}")

        return model

# --- Example Usage ---
model_path = "model_v14.onnx"  # Your converted model file
model = onnx.load(model_path)
print("Loaded model from", model_path)

# Apply the transformation to insert the SDP node.
transformation = InsertStreamingDataflowPartition()
model = transformation.apply(model)

# Optionally run shape inference to update shapes.
model = onnx.shape_inference.infer_shapes(model)

# Save the updated model.
new_model_path = "model.onnx"
onnx.save(model, new_model_path)
print("Saved updated model as", new_model_path)
