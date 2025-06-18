from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# --- INIT FLASK APP ---
app = Flask(__name__)
CORS(app)

# --- INIT FIREBASE ---
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- GLOBALS FOR MODEL & DATA ---
recipe_df = pd.DataFrame()
vectorizer = TfidfVectorizer()
nn = NearestNeighbors(metric='cosine', algorithm='brute')
X_ingredients = None

# --- LOAD AND PREPARE DATA ---
def fetch_recipes():
    recipes = db.collection('recipes').stream()
    data = []
    for r in recipes:
        d = r.to_dict()
        d['id'] = r.id
        data.append(d)
    return pd.DataFrame(data)

def extract_ingredient_names(ingredients):
    return ' '.join([item['name'] for item in ingredients if isinstance(item, dict) and 'name' in item])

def train_model():
    global recipe_df, vectorizer, nn, X_ingredients
    recipe_df = fetch_recipes()
    recipe_df['ingredients_text'] = recipe_df['ingredients'].apply(extract_ingredient_names)
    vectorizer = TfidfVectorizer()
    X_ingredients = vectorizer.fit_transform(recipe_df['ingredients_text'])
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(X_ingredients)

# Initial training
train_model()

# --- RECOMMENDATION ENDPOINT ---
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_input = request.get_json()
        ingredient_objects = user_input.get("ingredients", [])
        extracted_names = [item['name'] for item in ingredient_objects if isinstance(item, dict) and 'name' in item]

        if not extracted_names:
            return jsonify({"status": "error", "message": "No valid ingredients provided."}), 400

        user_text = ' '.join(extracted_names)
        vec = vectorizer.transform([user_text])
        distances, indices = nn.kneighbors(vec, n_neighbors=5)
        results = recipe_df.iloc[indices[0]]

        recommendations = []
        for _, row in results.iterrows():
            recommendations.append({
                "id": row['id'],
                "dish_name": row['dish_name'],
                "style": row.get('style', ''),
                "number_favourites": row.get('number_favourites', 0),
                "category": row.get('category', ''),
                "image_url": row.get('image_url', ''),
                "description": row.get('description', ''),
                "ingredients": row.get('ingredients', []),
                "cooking_steps": row.get('cooking_steps', [])
            })

        return jsonify({"status": "success", "recommendations": recommendations})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- RETRAIN ENDPOINT ---
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        train_model()
        return jsonify({"status": "success", "message": "Model retrained successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- MAIN ENTRY ---
if __name__ == '__main__':
    app.run()
