import json
import pandas as pd
import requests
from io import StringIO
from flask import Flask, request
from github import Github

app = Flask(__name__)

# Load the dataset from GitHub
access_token = "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"
g = Github(access_token)
repo = g.get_repo("archis-07/KNN_recommendation")
file_path = "Study_rooms_dataset_final.csv"
content = repo.get_contents(file_path)
data_url = content.download_url

response = requests.get(data_url)
data_str = response.content.decode("utf-8")
data = pd.read_csv(StringIO(data_str))

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Recommendation API!"

@app.route('/create_room', methods=['POST', 'GET'])
def create_room():
    global data  # Declare data as a global variable
    if request.method == 'POST':
        # Get input data from form
        room = request.form['room']
        domain = request.form['domain']
        age = request.form['age']
        reputation = request.form['reputation']

        new_room = pd.DataFrame({
            'Domain': [domain],
            'Age': [age],
            'Level of Study': [None],
            'Activity Level': [None],
            'Reputation': [reputation],
            'Accessibility': [None],
            'Moderation': [None],
            'Gender': [None],
            'Country': [None],
            'Recommended chat rooms': [room]
        })

        # Concatenate the new room DataFrame with the existing data
        data = pd.concat([data, new_room], ignore_index=True)

        # Save the updated dataset to GitHub
        csv_data = data.to_csv(index=False)
        repo.update_file(file_path, "Update dataset", csv_data, content.sha, branch="main")

        return "New room created and added to the dataset!"
    else:
        return """
        <form action="/create_room" method="POST">
            <label for="room">Room:</label><br>
            <input type="text" id="room" name="room"><br>
            <label for="domain">Domain:</label><br>
            <input type="text" id="domain" name="domain"><br>
            <label for="age">Age:</label><br>
            <input type="number" id="age" name="age"><br>
            <label for="reputation">Reputation:</label><br>
            <input type="number" id="reputation" name="reputation"><br><br>
            <input type="submit" value="Submit">
        </form>
        """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
