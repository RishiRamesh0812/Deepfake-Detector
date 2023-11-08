import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# loading the trained model
model = load_model("D:/Project DD/Deepfake Detector/model_version 2.h5")

# preprocessing the input
uploaded_img = image.load_img("D:/Project DD/images/24.jpeg",target_size=(224,224))
uploaded_img = image.img_to_array(uploaded_img)
uploaded_img = np.expand_dims(uploaded_img, axis=0)
uploaded_img /= 255.0

# predicting the model
prediction = model.predict(uploaded_img)

# Decide the class based on the threshold (0.5)
threshold = 0.9
if prediction[0][0] > threshold:
    result = "Real"
else:
    result = "Fake"

print("Prediction:", result)