import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/archis-07/KNN_recommendation/main/Study_rooms_dataset_final.csv")

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the domain column
domain_vectors = vectorizer.fit_transform(data['Domain'])

# Create and fit the Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
nn_model.fit(domain_vectors)

# Function to recommend chat rooms
def recommend_chat_rooms(domain, age, reputation):
    # Check if the entered domain is in the dataset
    if domain.lower() in data['Domain'].str.lower().unique():
        # Filter rooms with the entered domain and similar age group
        recommended_rooms = data[
            (data['Domain'].str.lower() == domain.lower()) &
            (data['Age'] >= age - 4) &
            (data['Age'] <= age + 4)
        ]

        # Sort by reputation and get top 5 rooms
        recommended_rooms = recommended_rooms.sort_values('Reputation', ascending=False)[:5]

        recommended_rooms = recommended_rooms[['Recommended chat rooms', 'Domain', 'Age', 'Reputation']].values.tolist()
    else:
        # Transform the user-entered domain
        user_domain = vectorizer.transform([domain])

        # Find the nearest neighbors
        distances, indices = nn_model.kneighbors(user_domain)

        # Get the recommended chat rooms (excluding the input)
        recommended_rooms = data.iloc[indices[0][1:]][['Recommended chat rooms', 'Domain', 'Age', 'Reputation']]

        # Filter rooms with similar age group
        recommended_rooms = recommended_rooms[
            (recommended_rooms['Age'] >= age - 4) & (recommended_rooms['Age'] <= age + 4)
        ]

        # Sort by reputation and get top 5 rooms
        recommended_rooms = recommended_rooms.sort_values('Reputation', ascending=False)
        recommended_rooms = recommended_rooms[['Recommended chat rooms', 'Domain', 'Age', 'Reputation']].values.tolist()

    return recommended_rooms

# Example usage of recommend_chat_rooms function
domain = input("Enter the domain: ")
age = int(input("Enter age: "))
reputation = float(input("Enter reputation: "))

recommendations = recommend_chat_rooms(domain, age, reputation)
print("Recommended Chat Rooms:")
for room in recommendations:
    print("Room:", room[0])
    print("Domain:", room[1])
    print("Age:", room[2])
    print("Reputation:", room[3])
    print()
