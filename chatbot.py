import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Sample training data (questions)
questions = [
    "What is Syed Abdullah proficient in?",
    "Where can you find Syed Abdullah's projects and contributions?",
    "What is Syed Abdullah currently working on?",
    "What are the main features of the blogging application project Syed Abdullah is working on?",
    "What AI and ML-based enhancements does the blogging application project include?",
    "What is Syed Abdullah's commitment in his projects?",
    "What type of projects has Syed Abdullah worked on previously?",
    "What does Syed Abdullah actively participate in outside of his work?",
    "What is one of Syed Abdullah's additional expertise beyond traditional software development?",
    "What are some of Syed Abdullah's hobbies outside of work?",
    "What type of developer is Syed Abdullah?",
    "Where can one find Syed Abdullah's GitHub profile?",
    "What is Syed Abdullah's primary focus in his current project?",
    "What are the authentication features included in Syed Abdullah's project?",
    "What are some AI-based enhancements in Syed Abdullah's project?",
    "What is Syed Abdullah known for in his projects?",
    "What technology has Syed Abdullah used in his projects?",
    "What is one of Syed Abdullah's interests outside of work?",
    "What kind of contributions does Syed Abdullah make outside of work?",
    "What is one example of a server project that Syed Abdullah has worked on?",
    "What is one example of a web application that Syed Abdullah has worked on?",
    "What are some of Syed Abdullah's skills?",
    "What areas of t    echnology does Syed Abdullah have extensive knowledge in?",
    "What is one of the AI enhancements in Syed Abdullah's project?",
    "What type of enhancements does Syed Abdullah's project include?",
    "What is Syed Abdullah's expertise in?",
    "What is Syed Abdullah's perspective on solving complex problems?",
    "What is Syed Abdullah known for besides his technical skills?",
    "What are some of Syed Abdullah's personal interests?",
    "What does Syed Abdullah like to do in his free time?",
    "What is Syed Abdullah passionate about?",
    "What type of applications has Syed Abdullah worked on?",
    "What type of projects has Syed Abdullah delivered?",
    "What kind of environment does Syed Abdullah thrive in?",
    "What are some of Syed Abdullah's personal hobbies?",
    "What are some of Syed Abdullah's professional skills?",
    "What is Syed Abdullah's perspective on innovation?",
    "What does Syed Abdullah enjoy doing outside of work?",
    "What is one area that Syed Abdullah explores beyond traditional software development?",
    "What type of projects does Syed Abdullah like to work on?",
    "What is one area where Syed Abdullah likes to push boundaries?",
    "What does Syed Abdullah enjoy from classic novels to the latest research papers?",
    "What is one way Syed Abdullah stays updated with the latest advancements?",
    "What is one thing that Syed Abdullah contributes to research communities?",
    "What is one thing Syed Abdullah contributes to open-source initiatives?",
    "What is one thing Syed Abdullah does to advance the field?",
    "What is one thing that Syed Abdullah possesses a keen understanding of?",
    "What is one technology that Syed Abdullah has worked with?",
    "What is one project that Syed Abdullah has successfully delivered?",
    "What is one platform that Syed Abdullah has worked with?",
    "What is one technology that Syed Abdullah is deeply involved in?",
    "What is one domain that Syed Abdullah's experience with blockchain benefits?",
    "What are some characteristics of Syed Abdullah as a team player?",
    "What is one thing that Syed Abdullah enjoys doing outdoors?",
    "What is one activity that Syed Abdullah enjoys?",
    "What is one thing that Syed Abdullah seeks inspiration from?",
    "What is one thing that Syed Abdullah enjoys in the natural world?",
    "What is one aspect of technology that Syed Abdullah is passionate about?",
    "What is one technology that Syed Abdullah integrates into his projects?",
    "What is one area that Syed Abdullah explores beyond traditional software development?",
    "What is one thing that Syed Abdullah actively participates in?",
    "What is one thing that Syed Abdullah contributes to outside of work?",
    "What is one project that Syed Abdullah has worked on previously?",
    "What is one technology that Syed Abdullah has experience with?",
    "What is one thing that Syed Abdullah enjoys doing outside of work?",
    "What is one area of expertise that Syed Abdullah has?",
    "What is one area that Syed Abdullah has knowledge in?",
    "What is one feature of Syed Abdullah's current project?",
    "What is one type of enhancement in Syed Abdullah's project?",
]
answers = [
    "MERN stack and Python",
    "https://github.com/Abdullaah-Ali",
    "A blogging application",
    "User authentication (login, signup, logout) using JWT, creating, viewing, and displaying blogs with pagination, and user profile view and edit functionality",
    "AI-powered spell checking, content summarization, a custom chatbot, and automatically generating SEO tags using NLP",
    "Syed Abdullah is committed to leveraging his expertise in ML and AI to build innovative solutions and improve user experiences.",
    "Syed Abdullah has worked on server projects like Neo NFT and sophisticated web applications powered by the MERN stack and Python.",
    "Outside of work, Syed Abdullah actively participates in reading, hiking, and photography.",
    "One of Syed Abdullah's additional expertise beyond traditional software development is blockchain technology.",
    "Some of Syed Abdullah's hobbies outside of work include reading, hiking, and photography.",
    "Syed Abdullah is a dedicated software engineer.",
    "Syed Abdullah's GitHub profile can be found at https://github.com/Abdullaah-Ali.",
    "Syed Abdullah's primary focus in his current project is a blogging application.",
    "The authentication features included in Syed Abdullah's project are user authentication (login, signup, logout) using JWT.",
    "Some AI-based enhancements in Syed Abdullah's project are: spell checks, content summarization, custom chatbot, and SEO tags using NLP.",
    "Syed Abdullah is known for leveraging his expertise in ML and AI to build innovative solutions and improve user experiences in his projects.",
    "Syed Abdullah has used technologies like the MERN stack, Python, and blockchain in his projects.",
    "One of Syed Abdullah's interests outside of work is reading.",
    "Outside of work, Syed Abdullah contributes by attending conferences, contributing to open-source initiatives, and participating in research communities.",
    "One example of a server project that Syed Abdullah has worked on is Neo NFT.",
    "One example of a web application that Syed Abdullah has worked on is a sophisticated web application powered by the MERN stack and Python.",
    "Some of Syed Abdullah's skills include proficiency in the MERN stack, Python, machine learning, and artificial intelligence.",
    "Syed Abdullah has extensive knowledge in areas like the MERN stack, Python, machine learning, and artificial intelligence.",
    "One of the AI enhancements in Syed Abdullah's project is spell checks.",
    "Syed Abdullah's project includes enhancements such as spell checks, content summarization, custom chatbot, and SEO tags using NLP.",
    "Syed Abdullah's expertise lies in leveraging his knowledge in ML and AI to build innovative solutions and improve user experiences in his projects.",
    "Syed Abdullah's perspective on solving complex problems is deeply involved in exploring the boundaries of technologies like machine learning and artificial intelligence.",
    "Besides his technical skills, Syed Abdullah is known for his effective communication skills and ability to mentor junior developers.",
    "Some of Syed Abdullah's personal interests include reading, hiking, and photography.",
    "In his free time, Syed Abdullah enjoys reading, hiking, and photography.",
    "Syed Abdullah is passionate about leveraging his expertise in ML and AI to build innovative solutions and improve user experiences in his projects.",
    "Syed Abdullah has worked on applications such as server projects like Neo NFT and sophisticated web applications powered by the MERN stack and Python.",
    "Syed Abdullah has delivered projects such as server projects like Neo NFT and sophisticated web applications powered by the MERN stack and Python.",
    "Syed Abdullah thrives in dynamic environments where creativity and innovation are valued.",
    "Some of Syed Abdullah's personal hobbies include reading, hiking, and photography.",
    "Some of Syed Abdullah's professional skills include proficiency in the MERN stack, Python, machine learning, and artificial intelligence.",
    "Syed Abdullah actively seeks to push the boundaries of what's possible with technology, always striving to innovate and create novel solutions.",
    "Outside of work, Syed Abdullah enjoys reading, hiking, and photography.",
    "One area that Syed Abdullah explores beyond traditional software development is blockchain technology.",
    "Syed Abdullah likes to work on projects that involve leveraging cutting-edge technologies like machine learning and artificial intelligence.",
    "One area where Syed Abdullah likes to push boundaries is in the integration of AI and ML into his projects.",
    "Syed Abdullah enjoys seeking inspiration from a wide range of literature, from classic novels to the latest research papers in AI and ML.",
    "One way Syed Abdullah stays updated with the latest advancements is by attending conferences and participating in research communities.",
    "One thing that Syed Abdullah contributes to research communities is by actively participating and sharing his expertise.",
    "One thing Syed Abdullah contributes to open-source initiatives is by sharing his projects and code on platforms like GitHub.",
    "One thing Syed Abdullah does to advance the field is by actively exploring and experimenting with cutting-edge technologies.",
    "One thing that Syed Abdullah possesses a keen understanding of is blockchain technology.",
    "One technology that Syed Abdullah has worked with is blockchain.",
    "One project that Syed Abdullah has successfully delivered is Neo NFT.",
    "One platform that Syed Abdullah has worked with is the MERN stack.",
    "One technology that Syed Abdullah is deeply involved in is artificial intelligence.",
    "One domain that Syed Abdullah's experience with blockchain benefits is understanding decentralized architecture.",
    "Syed Abdullah is known for his effective communication skills and ability to mentor junior developers, making him a collaborative team player.",
    "One thing that Syed Abdullah enjoys doing outdoors is hiking.",
    "One activity that Syed Abdullah enjoys is photography.",
    "One thing that Syed Abdullah actively seeks inspiration from is a wide range of literature, from classic novels to the latest research papers in AI and ML.",
    "One thing that Syed Abdullah enjoys in the natural world is the beauty of the natural world itself.",
    "One aspect of technology that Syed Abdullah is passionate about is leveraging his expertise in ML and AI to build innovative solutions and improve user experiences in his projects.",
    "One technology that Syed Abdullah integrates into his projects is machine learning.",
    "One area that Syed Abdullah explores beyond traditional software development is blockchain technology.",
    "One thing that Syed Abdullah actively participates in is attending conferences and participating in research communities.",
    "One thing that Syed Abdullah contributes to outside of work is by sharing his expertise in open-source initiatives and participating in research communities.",
    "One project that Syed Abdullah has worked on previously is Neo NFT.",
    "One technology that Syed Abdullah has experience with is blockchain.",
    "One thing that Syed Abdullah enjoys doing outside of work is reading.",
    "One area of expertise that Syed Abdullah has is in machine learning.",
    "One area that Syed Abdullah has knowledge in is the MERN stack.",
    "One feature of Syed Abdullah's current project is a blogging application.",
    "One type of enhancement in Syed Abdullah's project is AI-powered spell checking."
]
 # Encode answers
label_encoder = LabelEncoder()
encoded_answers = label_encoder.fit_transform(answers)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)
padded_sequences = pad_sequences(sequences, padding='post')

# Define the inputs
input_text = Input(shape=(padded_sequences.shape[1],))

# Define the embedding layer
embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=padded_sequences.shape[1])

# Get the embedded version of the input
embedded_input = embedding(input_text)

# Define the LSTM layers
lstm_layer = Bidirectional(LSTM(64, return_sequences=True))
lstm_output = lstm_layer(embedded_input)

# Apply GlobalMaxPooling1D to get the most relevant features
pooled_output = GlobalMaxPooling1D()(lstm_output)

# Define the outputs
output = Dense(len(label_encoder.classes_), activation='softmax')(pooled_output)

# Define the model
model = Model(inputs=input_text, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, encoded_answers, epochs=600, batch_size=200)

# Save the model
model.save('chatbot_model_enhanced_functional.keras')

# Save the tokenizer and label encoder
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
    
    
#real code