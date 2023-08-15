from flask import Flask, render_template, request
import joblib
import re
import preprocess_text as pt
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("xgboost_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")



# Preprocessing function
def preprocess_text(text):
    text = pt.convert_to_lowercase(text)
    text = pt.remove_whitespace(text)
    text = pt.re.sub('\n' , '', text) # converting text to one line
    text = pt.re.sub('\[.*?\]', '', text) # removing square brackets
    text = pt.remove_http(text)
    text = pt.remove_punctuation(text)
    text = pt.remove_html(text)
    text = pt.remove_emoji(text)
    text = pt.convert_acronyms(text)
    text = pt.convert_contractions(text)
    text = pt.remove_stopwords(text)
#     text = pyspellchecker(text)
    text = pt.text_lemmatizer(text) # text = text_stemmer(text)
    text = pt.discard_non_alpha(text)
    text = pt.keep_pos(text)
    text = pt.remove_additional_stopwords(text)
    return text

@app.route('/')
def index():
    # Sample product data
    products = [
        
    {"category": "Electronics", "description": "Wireless Noise-Canceling Headphones"},
    {"category": "Household", "description": "Stainless Steel Kitchen Utensil Set"},
    {"category": "Clothing & Accessories", "description": "Men's Casual T-Shirt"},
    {"category": "Books", "description": "Science Fiction Novel - The Galactic Odyssey"},
    {"category": "Electronics", "description": "Smartphone with Dual Cameras"},
    {"category": "Household", "description": "Set of Soft Bed Sheets"},
    {"category": "Clothing & Accessories", "description": "Women's Leather Handbag"},
    {"category": "Books", "description": "Mystery Thriller - The Enigma Code"},
    {"category": "Electronics", "description": "Smart Watch with Health Tracking"},
    {"category": "Household", "description": "Set of Glass Food Storage Containers"},
    {"category": "Clothing & Accessories", "description": "Kids' Colorful Backpack"},
    {"category": "Books", "description": "Fantasy Novel - The Kingdom's Quest"},
    {"category": "Electronics", "description": "Bluetooth Portable Speaker"},
    {"category": "Household", "description": "Ceramic Non-Stick Cookware Set"},
    {"category": "Clothing & Accessories", "description": "Classic Men's Leather Shoes"},
    {"category": "Books", "description": "Historical Fiction - The Time Traveler's Diary"},
    {"category": "Electronics", "description": "Gaming Console with High-Definition Graphics"},
    {"category": "Household", "description": "Robot Vacuum Cleaner"},
    {"category": "Clothing & Accessories", "description": "Women's Elegant Evening Dress"},
    {"category": "Books", "description": "Self-Help Guide - The Path to Success"}
]

    
    return render_template('index.html', products=products)


def nameOfLabel(number):
    if number == 0:
        return 'Electronics'
    elif number == 1:
        return 'Household'
    elif number == 2:
        return 'Books'
    elif number == 3:
        return 'Clothing & Accessories'
    
@app.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text')
    preprocessed_text = preprocess_text(text)  # Perform preprocessing
    text_vectorized = vectorizer.transform([preprocessed_text])
    predicted_label = model.predict(text_vectorized)[0]
    return render_template(
        'index.html', 
        prediction=f"Predicted category: {nameOfLabel(predicted_label)}",
        text=text)

if __name__ == '__main__':
    app.run(debug=True)