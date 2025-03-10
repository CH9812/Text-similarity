import pickle
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained model
with open("similarity_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load SentenceTransformer model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Text Similarity API is running!", 200

@app.route("/predict_similarity", methods=["POST"])
def predict_similarity():
    try:
        # Get JSON request data
        data = request.get_json()
        text1 = data.get("text1", "")
        text2 = data.get("text2", "")

        # Check if text is provided
        if not text1 or not text2:
            return jsonify({"error": "Both 'text1' and 'text2' are required"}), 400

        # Generate sentence embeddings
        embedding1 = sbert_model.encode([text1])
        embedding2 = sbert_model.encode([text2])

        # Compute cosine similarity
        cosine_sim_sbert = cosine_similarity(embedding1, embedding2)[0][0]

        # Prepare input features
        features = np.array([[cosine_sim_sbert, cosine_sim_sbert]])  # Using SBERT similarity twice

        # Predict similarity score using the trained model
        similarity_score = model.predict(features)[0]

        # Ensure output is between 0 and 1
        similarity_score = max(0, min(1, similarity_score))

        # Return response
        return jsonify({"similarity score": round(similarity_score, 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
