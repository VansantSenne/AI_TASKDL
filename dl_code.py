import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory

NUM_CLASSES = 5
IMG_SIZE = 64
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

# Function to create the model
def create_model():
    model = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGTH_FACTOR, WIDTH_FACTOR),
        layers.RandomZoom(0.2),
        layers.Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Streamlit UI
st.title("Image Classification with Streamlit")

# Create the model
model_new = create_model()

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img_array = image.img_to_array(image_display)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model_new.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    st.write(f"Prediction: Class {predicted_class}")

    # Display the class probabilities
    st.write("Class Probabilities:")
    for i in range(NUM_CLASSES):
        st.write(f"Class {i}: {predictions[0][i]:.4f}")

# Add other Streamlit components or features as needed...

# Add a sidebar with model summary
st.sidebar.header("Model Summary")
with st.sidebar.beta_expander("Click to show model summary"):
    model_new.summary()

# Add additional features or information as needed...

# Continue with the rest of your Streamlit code...
