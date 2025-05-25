from flask import Flask, render_template, make_response
from random import randrange
from recommender import get_recommendations
from db import add_reward

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend')
def recommend():
    return get_recommendations()


@app.route('/reward/<model>/<reward>', methods=['GET', 'POST'])
def reward(model: int, reward: int):
    add_reward(model, reward)
    return make_response({}, 204)


