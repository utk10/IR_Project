import spacy
import json
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score



# Load the JSON data from a file
with open('full_format_recipes.json', 'r') as file:
    data = json.load(file)

# Check if the data is a dictionary or a list
if isinstance(data, dict) and "root" in data:
    recipes = data["root"]
    recipes = [recipe for recipe in recipes if recipe.get("calories") is not None and recipe.get("rating") is not None]
elif isinstance(data, list):
    recipes = [recipe for recipe in data if recipe.get("calories") is not None and recipe.get("rating") is not None]
else:
    raise ValueError("Unexpected JSON structure")

nlp = spacy.load("en_core_web_sm")

# Add compound ingredients here
compound_ingredients = [
    "chicken breast", "garlic minced", "olive oil", "basil fresh", "tomato sauce",
    "ground pepper", "chicken thighs", "dried oregano", "fresh parsley", "fresh cilantro",
    "chicken stock", "vegetable stock", "lemon juice", "black pepper"
]

def normalize_ingredient_text(text):
    """Normalize ingredient text: lowercase, strip whitespace, remove punctuation."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        category = "Normal weight"
    elif 25 <= bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obese"
    return bmi, category

def filter_recipes_by_bmi(recipes, bmi_category):
    if bmi_category == "Underweight":
        min_calories = 500
        max_calories = None
    elif bmi_category == "Normal weight":
        min_calories = 300
        max_calories = 700
    elif bmi_category == "Overweight":
        min_calories = None
        max_calories = 500
    else:
        min_calories = None
        max_calories = 400

    filtered_recipes = []
    for recipe in recipes:
        calories = recipe.get("calories", 0)
        rating = recipe.get("rating")
        if calories is None or rating is None:
            continue
        if (min_calories is None or calories >= min_calories) and (max_calories is None or calories <= max_calories):
            filtered_recipes.append(recipe)

    return filtered_recipes

def process_ingredient_query(query):
    query = query.lower()
    query_ingredients = query.split(",")  # Split based on commas
    extracted_ingredients = []

    for ingredient in query_ingredients:
        ingredient = ingredient.strip()
        matched = False

        # Check for compound ingredients first
        for compound in compound_ingredients:
            if compound in ingredient:
                extracted_ingredients.append(compound)
                matched = True
                break

        # If no compound ingredient matched, tokenize
        if not matched:
            doc = nlp(ingredient)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
            extracted_ingredients.extend(tokens)

    # Remove duplicates and return processed ingredients
    return list(set(extracted_ingredients))

def normalize_ingredients(ingredients):
    normalized_ingredients = []
    for ingredient in ingredients:
        if ingredient in compound_ingredients:
            normalized_ingredients.append(ingredient)
        else:
            normalized_ingredients.append(normalize_ingredient_text(ingredient))
    return normalized_ingredients

def filter_recipes_by_ingredients(recipes, ingredients):
    filtered_recipes = []
    ingredients = normalize_ingredients(ingredients)

    for recipe in recipes:
        recipe_ingredients = []
        for ing in recipe.get("ingredients", []):
            normalized_ing = normalize_ingredient_text(ing)
            if normalized_ing in compound_ingredients:
                recipe_ingredients.append(normalized_ing)
            else:
                recipe_ingredients.append(normalized_ing)

        # Prioritize exact matches for compound ingredients
        if any(ingredient in recipe_ingredients for ingredient in ingredients):
            filtered_recipes.append(recipe)

    return filtered_recipes

def tokenize_categories(categories):
    tokens = []
    for category in categories:
        doc = nlp(category.lower())
        tokens.extend([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return set(tokens)

def filter_recipes_by_category(recipes, user_category):
    user_category_tokens = tokenize_categories([user_category])
    filtered_recipes = []

    for recipe in recipes:
        recipe_category_tokens = tokenize_categories(recipe.get("categories", []))
        if recipe_category_tokens & user_category_tokens:
            filtered_recipes.append(recipe)

    return filtered_recipes

def rank_recipes_by_similarity_vector_space(filtered_recipes, user_ingredients):
    user_ingredients_text = " ".join(user_ingredients)
    recipe_texts = [str(recipe['ingredients']) for recipe in filtered_recipes]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_ingredients_text] + recipe_texts)

    ingredient_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    ratings = [recipe.get('rating', 0) if recipe.get('rating') is not None else 0 for recipe in filtered_recipes]
    max_rating = max(ratings) if ratings else 1
    normalized_ratings = [rating / max_rating for rating in ratings]

    recipe_vectors = [
        [ingredient_similarities[i], normalized_ratings[i]]
        for i in range(len(filtered_recipes))
    ]
    user_vector = [1, 1]

    vector_space_similarities = cosine_similarity([user_vector], recipe_vectors).flatten()

    for i, recipe in enumerate(filtered_recipes):
        recipe['vector_space_similarity'] = vector_space_similarities[i]

    unique_recipes = {recipe['title']: recipe for recipe in filtered_recipes}
    top_recipes = sorted(unique_recipes.values(), key=lambda x: x['vector_space_similarity'], reverse=True)[:10]

    return top_recipes

def plot_bmi_scale(bmi):
    categories = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
    category_bmi_ranges = [0, 18.5, 24.9, 29.9, 40]
    fig, ax = plt.subplots(figsize=(10, 2))

    for i in range(len(category_bmi_ranges) - 1):
        ax.plot([category_bmi_ranges[i], category_bmi_ranges[i + 1]], [0, 0], color='black', lw=2)

    for i, category in enumerate(categories):
        ax.text((category_bmi_ranges[i] + category_bmi_ranges[i + 1]) / 2, 0.1, category, ha='center', va='bottom')

    ax.plot([bmi, bmi], [-0.1, 0.1], color='red', lw=3)
    ax.annotate(f'{bmi:.2f}', (bmi, -0.2), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
    ax.set_yticks([])
    ax.set_xticks(category_bmi_ranges[:-1])
    ax.set_xticklabels([f'{category_bmi_ranges[i]}-{category_bmi_ranges[i+1]}' for i in range(len(categories))])
    ax.set_xlim([0, 40])
    ax.set_ylim([-0.3, 0.3])
    ax.set_title("BMI Category Scale")
    st.pyplot(fig)

# Main function
def main():
    st.title("Healthy Recipe Finder")
    st.write("This application helps you find recipes based on your BMI, available ingredients, and category preference.")

    # Input for BMI calculation
    height = st.number_input("Enter your height in cm:", min_value=50, max_value=300, value=170)
    weight = st.number_input("Enter your weight in kg:", min_value=20, max_value=500, value=70)

    # Calculate BMI and determine category
    bmi, bmi_category = calculate_bmi(height, weight)
    st.write(f"BMI: {bmi:.2f}, Category: {bmi_category}")

    # Plot BMI scale (optional)
    plot_bmi_scale(bmi)

    # Ingredient query input
    user_query = st.text_input("Enter ingredients you have (comma separated):{Example: I have chicken,olive oil,salt}")
    if user_query:
        # Process user query for ingredients
        user_ingredients = process_ingredient_query(user_query)
        st.write(f"Ingredients extracted from your query: {user_ingredients}")

        # Filter recipes based on BMI category
        filtered_recipes_by_bmi = filter_recipes_by_bmi(data, bmi_category)
        filtered_recipes_2000 = filtered_recipes_by_bmi[:2000]

        # Filter recipes further by ingredients
        filtered_recipes_by_ingredients = filter_recipes_by_ingredients(filtered_recipes_2000, user_ingredients)

        # Category selection (applied after ingredient filtering)
        category = st.selectbox(
            "Select a recipe category (optional):",
            ["None", "Vegetarian", "Non-Veg", "Soup", "Cookie", "Kid-Friendly", "Breakfast", "Dinner", "Lunch"]
        )

        # Filter recipes by selected category if applicable
        if category != "None":
            filtered_recipes_by_ingredients = filter_recipes_by_category(filtered_recipes_by_ingredients, category)

        # Rank recipes using vector space similarity
        top_recipes = rank_recipes_by_similarity_vector_space(filtered_recipes_by_ingredients, user_ingredients)

        # Display top recipes
        st.write("Top 10 recipes based on ingredient match and similarity score:")
        for recipe in top_recipes:
            with st.expander(recipe['title']):
                st.write(f"**Rating:** {recipe.get('rating', 'N/A')} ⭐")
                st.write(f"**Calories:** {recipe.get('calories', 'N/A')} kcal")
                st.write(f"**Similarity Score:** {recipe.get('vector_space_similarity', 'N/A') * 100:.2f}%")
                st.write("**Ingredients:**")
                st.write("\n".join(f"- {ing}" for ing in recipe.get("ingredients", [])))
                st.write("**Directions:**")
                st.write("\n".join(f"{i + 1}. {direction}" for i, direction in enumerate(recipe.get("directions", []))))

if __name__ == "__main__":
    main()
