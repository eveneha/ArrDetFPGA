import onnx

def remove_extra_sdp_nodes(model):
    # Collect all nodes that are StreamingDataflowPartition
    sdp_nodes = [node for node in model.graph.node if node.op_type == "StreamingDataflowPartition"]
    if len(sdp_nodes) > 1:
        print(f"Found {len(sdp_nodes)} StreamingDataflowPartition nodes; removing extra ones.")
        # Keep only the first one, remove the rest.
        nodes_to_keep = {sdp_nodes[0]}
        new_node_list = []
        for node in model.graph.node:
            if node.op_type == "StreamingDataflowPartition":
                if node in nodes_to_keep:
                    new_node_list.append(node)
                else:
                    print(f"Removing node: {node.name or '<anonymous>'}")
            else:
                new_node_list.append(node)
        model.graph.node[:] = new_node_list
    else:
        print("Exactly one StreamingDataflowPartition node found; no removal needed.")
    return model

# Load your ONNX model (adjust the path as needed)
model_path = "model.onnx"
model = onnx.load(model_path)

# Remove extra StreamingDataflowPartition nodes
model = remove_extra_sdp_nodes(model)

# Save the updated model
onnx.save(model, "model_merged.onnx")
print("Saved merged model as model_merged.onnx")
