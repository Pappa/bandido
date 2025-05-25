from flask import Flask, render_template, make_response
from random import randrange
from recommender import get_recommendations
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', model=randrange(10), reward=randrange(10))


@app.route('/recommend')
def recommend():
    return get_recommendations()


@app.route('/reward/<model>/<reward>', methods=['GET', 'POST'])
def reward(model: int, reward: int):
    print(model, reward)
    return make_response({}, 204)
