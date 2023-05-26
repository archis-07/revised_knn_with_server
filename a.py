import json
from flask import Flask, request
from ml_model import recommend_chat_rooms

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Recommendation API!"

@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            # Get input data from JSON request
            data = request.json
            domain = data['domain']
            level_of_study = data['level_of_study']
            age = data['age']
            reputation = data['reputation']
        else:
            # Get input data from form data
            domain = request.form['domain']
            level_of_study = request.form['level_of_study']
            age = int(request.form['age'])
            reputation = float(request.form['reputation'])

        # Get recommendations using your recommendation function
        recommendations =recommend_chat_rooms(domain, level_of_study, age, reputation)  # Replace with your actual recommendations

        # Serialize the recommendations to JSON
        response = json.dumps({"recommendations": recommendations})

        # Return the JSON response
        return response
    else:
        # Return a form or webpage to enter the input
        return """
        <form action="/recommend" method="POST">
            <label for="domain">Domain:</label><br>
            <input type="text" id="domain" name="domain"><br>
            <label for="level_of_study">Level of Study:</label><br>
            <input type="text" id="level_of_study" name="level_of_study"><br>
            <label for="age">Age:</label><br>
            <input type="number" id="age" name="age"><br>
            <label for="reputation">Reputation:</label><br>
            <input type="number" id="reputation" name="reputation"><br><br>
            <input type="submit" value="Submit">
        </form>
        """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
