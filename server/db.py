import sqlite3

db = sqlite3.connect('rec.db', check_same_thread=False)

with open('schema.sql') as f:
    db.executescript(f.read())


def add_reward(model: int, reward: int):
    print(model, reward)
    db.execute('INSERT INTO rewards (model, reward) VALUES (?, ?)', (model, reward))
    db.commit()


