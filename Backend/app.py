from flask import Flask, request, jsonify
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from flask_cors import CORS

class GCNClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=2):  # ✅ 384 instead of 768
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Classification layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # ✅ Dropout again
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # ✅ Apply dropout after activation

        # Aggregate at the graph level
        x = global_mean_pool(x, batch)  # ✅ Pooling for graph classification

        # Final classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # ✅ Log softmax for classification

# Load the trained model
model_for_testing = GCNClassifier(input_dim=384, hidden_dim=128, output_dim=2)
model_path = "gcn_email_classifier.pth"
model_for_testing.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Load weights
model_for_testing.eval()

# Load the function to convert text into a PyG graph
from graph_module import create_word_graph  # Update with actual module

app = Flask(__name__)
CORS(app, resources={r"/classify": {"origins": "http://localhost:5173"}})

@app.route("/classify", methods=["POST"])
def classify_email():
    data = request.json
    email_text = data.get("email", "")

    if not email_text:
        return jsonify({"error": "No email text provided"}), 400

    # Convert text to a PyG graph
    email_graph = create_word_graph(email_text, window_size=3)
    email_graph = email_graph.to("cpu")

    # Make prediction
    with torch.no_grad():
        output = model_for_testing(email_graph.x, email_graph.edge_index, torch.tensor([0] * email_graph.x.shape[0]))
        prediction = torch.argmax(output, dim=1).item()

    label_map = {0: "Legitimate Email", 1: "Phishing Email"}
    return jsonify({"prediction": label_map[prediction]})

if __name__ == "__main__":
    app.run(debug=True)