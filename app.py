from flask import Flask, request, jsonify
import pickle
import string
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

# Flask app
app = Flask(__name__)

# Load RNN model and tokenizer
with open("w2v_lemma_rnn_model.pkl", "rb") as file:
    model = pickle.load(file)  # Load your RNN model
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)  # Load the tokenizer

# SpaCy model for lemmatization
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

# Constants for padding
max_len = 15  # Adjust based on training
max_vocab_size = 5000  # Matches tokenizer vocab size

# Preprocessing function
def clean_token_lemmatize(texts, batch_size=1000):
    """
    Clean, tokenize, and lemmatize input texts.
    """
    cleaned_texts = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        cleaned_tokens = [
            token.lemma_.lower() for token in doc
            if token.is_alpha and not token.is_stop and not token.like_num and token.text not in string.punctuation
        ]
        cleaned_texts.append(" ".join(cleaned_tokens))
    return cleaned_texts

def preprocess_input(text):
    """
    Full preprocessing pipeline: Clean, tokenize, and pad input text.
    """
    # Clean and lemmatize the input
    cleaned_text = clean_token_lemmatize([text])
    # Tokenize and pad the cleaned text
    sequences = tokenizer.texts_to_sequences(cleaned_text)
    padded_sequence = pad_sequences(sequences, maxlen=max_len)
    return padded_sequence

# Prediction endpoint
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Get text from query parameters if provided
        text = request.args.get("text", None)
        if not text:
            # Render the input form if no query parameter is present
            return '''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Text Prediction</title>
                </head>
                <body>
                    <h1>Enter Text for Prediction</h1>
                    <form action="/predict" method="get">
                        <label for="text">Enter your text:</label><br><br>
                        <input type="text" id="text" name="text" placeholder="Type here..." style="width:300px;"><br><br>
                        <input type="submit" value="Submit">
                    </form>
                </body>
                </html>
            '''
        # If text is provided as a query parameter, process and predict
        try:
            processed_input = preprocess_input(text)
            prediction = model.predict(processed_input)
            binary_result = (prediction > 0.5).astype(int)

            message = "This tweet is negative." if binary_result == 1 else "This tweet is not negative."

            # Render the prediction result as HTML
            return f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Prediction Result</title>
                </head>
                <body>
                    <h1>Prediction Result</h1>
                    <p><strong>Input Text:</strong> {text}</p>
                    <p><strong>Prediction:</strong> {message}</p>
                    <a href="/predict">Try Again</a>
                </body>
                </html>
            '''
        except Exception as e:
            # Display error in HTML
            return f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error</title>
                </head>
                <body>
                    <h1>Error</h1>
                    <p style="color:red;">{str(e)}</p>
                    <a href="/predict">Go Back</a>
                </body>
                </html>
            '''
    elif request.method == 'POST':
        # Handle JSON input in POST request
        try:
            input_data = request.get_json()
            text = input_data.get("text", "")
            if not text:
                return jsonify({"error": "No input text provided"}), 400

            # Preprocess and predict
            processed_input = preprocess_input(text)
            prediction = model.predict(processed_input)
            binary_result = (prediction > 0.5).astype(int)

            # Prepare and return JSON response
            message = "This tweet is negative." if binary_result == 1 else "This tweet is not negative."
            return jsonify({"prediction": message})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

            
# Run Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
