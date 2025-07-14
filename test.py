# hello_fl.py
import tensorflow_federated as tff
import pandas as pd
@tff.federated_computation
def say_hello():
    return 'Hello FL!'

print(say_hello())
