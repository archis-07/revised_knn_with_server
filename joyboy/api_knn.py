import json
from flask import Flask, request
from knn_be_final import recommend_chat_rooms

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
            age = int(data['age'])
            reputation = float(data['reputation'])
        else:
            # Get input data from form data
            domain = request.form['domain']
            age = int(request.form['age'])
            reputation = float(request.form['reputation'])

        # Get recommendations using your recommendation function
        recommendations = recommend_chat_rooms(domain, age, reputation)  # Replace with your actual recommendations

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
            <label for="age">Age:</label><br>
            <input type="number" id="age" name="age"><br>
            <label for="reputation">Reputation:</label><br>
            <input type="number" id="reputation" name="reputation"><br><br>
            <input type="submit" value="Submit">
        </form>
        """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
