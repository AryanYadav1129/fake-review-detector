from flask import Flask, request, render_template
import joblib

# Load the model and vectorizer
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Ensure this is the same vectorizer used during training

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the form
    review_text = request.form['review']
    
    # Transform the review text using the loaded vectorizer
    review_vectorized = vectorizer.transform([review_text])
    
    # Get the probabilities of the review being real or fake
    prob = model.predict_proba(review_vectorized)[0]
    
    # Print probabilities for debugging
    print(f"Probabilities: {prob}")  # Example: [0.2, 0.8] -> 80% chance it's fake
    
    # Extract the probability of being fake (second index in prob)
    fake_prob = prob[1]
    
    # Lower the threshold for classification
    threshold = 0.6  # Lowered threshold for testing
    if fake_prob > threshold:
        result = "Fake Review"
    else:
        result = "Real Review"
    
    # Render the template and pass review text and result for display
    return render_template('index.html', review=review_text, result=result)

if __name__ == "__main__":
    app.run(debug=True)
