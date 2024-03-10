from flask import Flask, render_template, request
import pickle as pk
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# creating my flask application
app = Flask(__name__)

# loading model via pickle
model = pk.load(open('model.pkl', 'rb'))

# simple GET endpoint in case user request the index page
@app.route('/', methods=['GET'])
def get_index():
    return render_template('index.html')


# this POST endpoint gets image from user and predicts it with deep learning model
@app.route('/', methods=['POST'])
def predict():
    imageFile = request.files['imageFile']
    imagePath = "./static/predicted_images/" + imageFile.filename
    imagePath2 = "/static/predicted_images/" + imageFile.filename
    print(imagePath)
    imageFile.save(imagePath)

    image_to_predict = load_img(imagePath, target_size=(224, 224))
    image_tensor = img_to_array(image_to_predict)
    image_tensor = np.expand_dims(image_tensor, axis=0)

    result = model.predict(image_tensor)
    classes_x=np.argmax(result,axis=1)

    print("RESULT ", result)
    print("CLASSE ", classes_x[0])

    negative = "negative"
    positive = "positive"
    
    print("ImagePath ", imagePath2)

    if classes_x[0] == 0:
        return render_template('index.html', negative=negative, imagePath2=imagePath2)
    else:
        return render_template('index.html', positive=positive, imagePath2=imagePath2)


# starting point of my web application
if __name__ == '__main__':
    app.run(port=3000, debug=True)    
