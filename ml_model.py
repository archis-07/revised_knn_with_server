import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (currently from the Google Drive)
data = pd.read_csv("https://raw.githubusercontent.com/archis-07/KNN_recommendation/main/Study_rooms_dataset_final.csv")

# Preprocessing the dataset (preprocessing already done in Excel)
X = data[['Domain', 'Level of Study', 'Age', 'Reputation']]
y = data['Recommended chat rooms']

# Perform one-hot encoding for categorical variables
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = onehot_encoder.fit_transform(X[['Domain', 'Level of Study']])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Age', 'Reputation']])

# Concatenate the encoded categorical features and scaled numerical features
X_processed = pd.concat([pd.DataFrame(X_encoded), pd.DataFrame(X_scaled)], axis=1)

# Create and fit the Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
nn_model.fit(X_processed)

# Function to recommend chat rooms
def recommend_chat_rooms(domain, level_of_study, age, reputation):
    # Check if the entered domain is in the dataset
    if domain in data['Domain'].unique():
        # Preprocess input
        input_data = pd.DataFrame({'Domain': [domain],
                                   'Level of Study': [level_of_study],
                                   'Age': [age],
                                   'Reputation': [reputation]})

        # Perform one-hot encoding for categorical variables
        input_encoded = pd.DataFrame(onehot_encoder.transform(input_data[['Domain', 'Level of Study']]))

        # Scale numerical features
        input_scaled = pd.DataFrame(scaler.transform(input_data[['Age', 'Reputation']]))

        # Concatenate the encoded categorical features and scaled numerical features
        input_processed = pd.concat([input_encoded, input_scaled], axis=1)

        # Find the nearest neighbors
        distances, indices = nn_model.kneighbors(input_processed)

        # Get the recommended chat rooms (excluding the input)
        recommended_rooms = data.iloc[indices[0][1:]]['Recommended chat rooms'].values.tolist()

    else:
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Fit and transform the domain column
        domain_vectors = vectorizer.fit_transform(data['Domain'])

        # Transform the user-entered domain
        user_domain = vectorizer.transform([domain])

        # Calculate cosine similarity between user domain and dataset domains
        similarities = cosine_similarity(user_domain, domain_vectors)

        # Get indices of top similar domains
        top_indices = similarities.argsort()[0][-20:][::-1]

        # Get recommended chat rooms based on similar domains
        similar_rooms = data.iloc[top_indices]['Recommended chat rooms'].values.tolist()

        recommended_rooms = []

        # Check if the similar rooms are related to the entered domain
        for room in similar_rooms:
            if domain.lower() in room.lower():
                recommended_rooms.append(room)

            if len(recommended_rooms) >= 10:
                break

    return recommended_rooms[:10]

# Example usage of recommend_chat_rooms function
'''
domain = input("Enter the domain: ")
level_of_study = input("Enter the level of study: ")
age = int(input("Enter age: "))
reputation = float(input("Enter reputation: "))

recommendations = recommend_chat_rooms(domain, level_of_study, age, reputation)
print("Recommended Chat Rooms:", recommendations)
'''
