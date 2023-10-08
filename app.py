import tensorflow as tf
from tensorflow.keras import layers
import os
import streamlit as st
import http
from utils import create_model, load_and_prep_image, class_names


def get_model(model_weights_path):
    # Recreate the model architecture and recompile
    model = create_model()
    model.compile(loss="sparse_categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])

    # Unfreeze all the layers in the model
    for layer in model.layers:
        layer.trainable = True # set all layers to trainable
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # make sure loaded model is using mixed precision dtype_policy ("mixed_float16")
    # Load the weight (Faster than loading the whole model)
    model.load_weights(model_weights_path)
    return model

def model_prediction(image, model):
    img = load_and_prep_image(image, scale=False) 
    img = tf.expand_dims(img, axis=0)
    pred_prob = model.predict(img)
    pred_conf = tf.reduce_max(pred_prob[0]) 
    pred_class = class_names[pred_prob.argmax()] 
    return pred_class, pred_conf

# Get the model
model = get_model("model/efficientnetb0_fine_tuned_101_classes_all_data_weights.h5")

# Main Body
def main():
    st.set_page_config(
        page_title="Food Vision",
        page_icon="🍔",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Define the web app title and its description
    html_code = """
    <h1 style='text-align: center; color: LightGray;'>Food Vision 🍔📷</h1>
    <p style='margin-bottom: 0px; margin-top: 0px; text-align: center;'>Computer vision web app to identify food in a image.</p>
    <p style='margin-top: 0px; margin-top: 0px; text-align: center;'>Simply upload a photo of food and Food Vision will tell you what it is.</p>
    <hr style='margin-top: 0; margin-bottom: 0;'>
    """

    # Display the HTML code on the Streamlit page
    st.markdown(html_code, unsafe_allow_html=True)
    
    # Get the image
    file = st.file_uploader(label="Upload an image of food.",
                            type=["jpg", "jpeg", "png"])


    # Make sure someone is uploading an image file
    if not file:
        st.warning("Please upload an image")
        st.stop()
    else:
        image = file.read() 
        st.image(image, use_column_width=True)
        pred_button = st.button("Predict")

    # When the pred_button is pressed
    if pred_button:
        with st.spinner('Wait for prediction....'):
            pred_class, pred_conf = model_prediction(image, model)
        
        st.success(f'Prediction : {pred_class}')
        st.success(f'Confidence : {pred_conf*100:.2f}%')


if __name__ == '__main__':
    main()