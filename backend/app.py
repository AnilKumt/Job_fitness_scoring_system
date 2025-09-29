from flask import Flask
from markupsafe import escape
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)


@app.route('/user/<username>')
def show_user_profile(username):
    return f'User {escape(username)}'

if __name__ == '__main__':
    app.run(debug=True)