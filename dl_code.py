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

# Define category names
category_names = ["Beaches", "Cities", "Forests", "Mountains", "Plain fields"]

# Function to create and train the model
def create_and_train_model(epochs):
    # Load or create your dataset
    # ...

    # Define and compile the model
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

    # Train the model on your dataset
    batch_size = 32
    image_size = (64, 64)
    validation_split = 0.2

    # Assuming you have a dataset directory structure similar to the one you used for training
    training_set = image_dataset_from_directory(
        directory='downloads/train',
        labels='inferred',
        subset='training',
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        label_mode='categorical',
        seed=42
    )
    
    validation_set = image_dataset_from_directory(
        directory='downloads/train',
        labels='inferred',
        subset='validation',
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        label_mode='categorical',
        seed=42
    )

    history = model.fit(training_set, validation_data=validation_set, epochs=epochs)

    return model, history

# Streamlit UI
st.title("Image Classification with Streamlit")
st.write("In this application you can train a model to classify images of landscapes with deeplearning.")
st.write("Below you can see the distribution of images used to train and test our model.")
image = Image.open("eda.JPG")
st.image(image, caption="eda", use_column_width=True)
# Initialize model_new outside the if block
model_new = None

num_epochs = st.slider("Selecteer het aantal epochs", min_value=1, max_value=50, value=20)

train_button = st.button("Train Model")

if train_button:
    st.text(f"Training the model for {num_epochs} epochs. This might take some time...")
    model_new, training_history = create_and_train_model(num_epochs)
    st.text("Training completed!")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded)
    st.image(image_display, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img_array = image.img_to_array(image_display)
    img_array = np.expand_dims(img_array, axis=0)

    # Check if model_new is not None before making predictions
    if model_new is not None:
        # Make predictions
        predictions = model_new.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = category_names[predicted_class_index]

        st.write(f"Prediction: {predicted_class_name}")
        
   

# Add other Streamlit components or features as needed...
