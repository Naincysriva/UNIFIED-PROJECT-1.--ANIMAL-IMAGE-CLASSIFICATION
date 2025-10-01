import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = load_model("animal_model.h5")

# Load class labels from dataset
datagen = ImageDataGenerator(rescale=1./255)
dummy_data = datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical'
)

# Create label map
class_indices = dummy_data.class_indices
classes = dict((v, k) for k, v in class_indices.items())

# Load test image
img_path = "test_image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
pred = model.predict(img_array)
confidence = np.max(pred)
predicted_index = np.argmax(pred)

# Threshold for unknown
if confidence < 0.60:
    print(f"❌ Animal Not Found (Confidence: {confidence:.2f})")
else:
    predicted_class = classes[predicted_index]
    print(f"✅ Predicted Animal: {predicted_class} (Confidence: {confidence:.2f})")
