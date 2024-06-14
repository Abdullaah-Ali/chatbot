import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the trained model
model = load_model('chatbot_model_enhanced_functional.keras')

# Function to predict answer
def predict_answer(text):
    # Tokenize and pad input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=15, padding='post')

    # Predict probabilities for each answer class
    predicted_probabilities = model.predict(padded_sequence)

    # Get the predicted answer class
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

    return predicted_class

# Main loop to chat with the bot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    else:
        response = predict_answer(user_input)
        print("Bot:", response)
################################################################