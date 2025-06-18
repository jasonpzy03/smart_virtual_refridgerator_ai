import firebase_admin
from firebase_admin import credentials, firestore
import json

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load JSON data
with open("recipes.json", "r") as f:
    recipes = json.load(f)

# Add each recipe to the 'recipes' collection
for recipe in recipes:
    # You can use `add()` to let Firestore auto-generate an ID,
    # or `document(recipe["dish_name"])` to set your own (e.g., to avoid duplicates)
    db.collection("recipes").add(recipe)
    # To avoid duplicates on rerun, you can do:
    # db.collection("recipes").document(recipe["dish_name"]).set(recipe, merge=True)
