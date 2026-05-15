from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained model
model = MobileNet(weights='imagenet')

# Load sample image
img = image.load_img(
    r"D:\Decodelabs Internship\cat.jpg",
    target_size=(224,224) )

# Convert image to array
img_array = image.img_to_array(img)

# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# Preprocess image
img_array = preprocess_input(img_array)

# Perform recognition
prediction = model.predict(img_array)

# Display output clearly
result = decode_predictions(prediction, top=1)

print("Prediction:")
print(result[0][0][1])