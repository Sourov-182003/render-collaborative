from flask import Flask, request, jsonify, render_template
import pickle
from surprise import SVD

# Load saved models and data
print("Loading model and data...")
with open("svd_model.pkl", "rb") as f:
    svd_algo = pickle.load(f)

with open("user_item.pkl", "rb") as f:
    user_item = pickle.load(f)

with open("product_names.pkl", "rb") as f:
    product_names = pickle.load(f)

with open("product_aisles.pkl", "rb") as f:
    product_aisles = pickle.load(f)

app = Flask(__name__)


@app.route("/")
def home():
    """Render the home page with the recommendation UI"""
    return render_template("index.html")


@app.route("/recommend", methods=["GET"])
def recommend():
    """Get top recommended items for a user"""
    try:
        user_id = int(request.args.get("user_id", 1))
        n_recommendations = int(request.args.get("n", 10))

        interacted = set(user_item.get(user_id, {}))
        unseen = list(set(product_names.keys()) - interacted)

        predictions = [(pid, svd_algo.predict(user_id, pid).est) for pid in unseen]
        top_n = sorted(predictions, key=lambda x: -x[1])[:n_recommendations]

        result = [{"product": product_names[pid], "rating": round(rating, 2)}
                  for pid, rating in top_n if pid in product_names]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/recommend/aisle", methods=["GET"])
def recommend_aisle():
    """Get recommended items for a user in a specific aisle"""
    try:
        user_id = int(request.args.get("user_id", 1))
        aisle_name = request.args.get("aisle", "cookies cakes").strip().lower()
        n_recommendations = int(request.args.get("n", 10))

        aisle_products = {pid for pid, aisle in product_aisles.items() if aisle.lower() == aisle_name}
        if not aisle_products:
            return jsonify([])

        interacted = set(user_item.get(user_id, {}))
        unseen = list(aisle_products - interacted)

        predictions = [(pid, svd_algo.predict(user_id, pid).est) for pid in unseen]
        top_n = sorted(predictions, key=lambda x: -x[1])[:n_recommendations]

        result = [{"product": product_names[pid], "aisle": aisle_name, "rating": round(rating, 2)}
                  for pid, rating in top_n if pid in product_names]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
