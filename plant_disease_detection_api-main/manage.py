from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('./model')
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the form
        file = request.files['file']

        # Read the image file and convert it to RGB format
        img = Image.open(file.stream).convert('RGB')

        # Resize the image to match the input size of the model
        img = img.resize((256, 256))

        # Preprocess the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Perform prediction
        predictions = model.predict(img_array)

        # Retrieve the predicted class name
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]

        return render_template('index.html', prediction=predicted_class_name)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
