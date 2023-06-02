import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/archis-07/KNN_recommendation/main/Study_rooms_dataset_final.csv")

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
        recommended_rooms = recommended_rooms.sort_values('Reputation', ascending=False)[:10]

        recommended_rooms = recommended_rooms[['Recommended chat rooms', 'Domain', 'Age', 'Reputation']].values.tolist()
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
        similar_rooms = data.iloc[top_indices][['Recommended chat rooms', 'Domain', 'Age', 'Reputation']]

        # Filter rooms with similar age group
        similar_rooms = similar_rooms[
            (similar_rooms['Age'] >= age - 4) & (similar_rooms['Age'] <= age + 4)
        ]

        # Sort by reputation and get top 5 rooms
        recommended_rooms = similar_rooms.sort_values('Reputation', ascending=False)[:5]
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
