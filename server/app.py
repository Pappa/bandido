from flask import Flask, render_template, make_response

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend')
def recommend():
    return {}


@app.route('/reward', methods=['GET', 'POST'])
def reward():
    return make_response({}, 204)
