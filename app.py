import os
import io
import numpy as np
from PIL import Image
import base64

import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras import backend as K

from flask import Flask, request, redirect, jsonify, render_template

app = Flask(__name__)
model = None
graph = None

def load_model():
    global model
    global graph
    model = keras.models.load_model("sample.h5")
    graph = K.get_session().graph

load_model()

@app.route('/', methods=['GET', 'POST'])
def post_comment():
    data = {"success": False}
    if request.method == 'POST':
        if request.form.get('comment'):
            # read the base64 encoded string
            comment_string = request.form.get('comment')
            # Get the tensorflow default graph
            global graph
            with graph.as_default():

                # Use the model to make a prediction
                print(comment_string);
                classify_comment = model.predict_classes(comment_string)[0]
                data["prediction"] = str(classify_comment)

                # indicate that the request was a success
                data["success"] = True

        return jsonify(data)
    return render_template("index.html")

if __name__ == "__main__":
    load_model()
    app.run()
