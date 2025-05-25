from random import randrange

def get_recommendations():
    return {
        "model": randrange(10),
        "item": randrange(10)
    }