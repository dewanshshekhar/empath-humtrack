from flask import Flask, render_template, request, jsonify

# Create a Flask application instance
# __name__ helps Flask determine the root path of the application
app = Flask(__name__)

# The route() decorator tells Flask what URL should trigger the function
@app.route("/")
def index():
    return render_template("index.html")

# This block ensures the development server runs only when the script is executed directly
if __name__ == "__main__":
    # Run the app. By default, it runs on http://127.0.0.1:5000/
    app.run(debug=True)