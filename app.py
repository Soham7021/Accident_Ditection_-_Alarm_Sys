import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import tempfile
import pygame
from keras.models import model_from_json


class AccidentDetectionModel(object):

    class_nums = ['Accident', "No Accident"]

    def __init__(self, model_json_file, model_weights_file):
        
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        self.preds = self.loaded_model.predict(img)
        return AccidentDetectionModel.class_nums[np.argmax(self.preds)], self.preds

model = AccidentDetectionModel("model.json", 'model_weights.weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def predict_accident(img):
    img_resized = cv2.resize(img, (250, 250)) 
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  
    preds = model.loaded_model.predict(img_rgb[np.newaxis, :, :, :])
    class_nums = ['Accident', 'No Accident']
    pred_class = class_nums[np.argmax(preds)]
    return pred_class, preds

def main():
    st.title('Accident Detection')

    
    detect_live_camera = st.checkbox("Detect from Live Camera")

    if not detect_live_camera:
        uploaded_file = st.file_uploader("Upload an image or video...", type=["mp4", "avi"])
        if uploaded_file is not None:
            
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_file.close()

            
            cap = cv2.VideoCapture(temp_file.name)

            
            pygame.mixer.init()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("End of video.")
                    break

            
                print("Original Frame Shape:", frame.shape)

                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                
                pred, prob = predict_accident(frame_rgb)

                
                if prob[0][0] > 0.95:
                    st.image(frame_rgb, caption=f'Prediction: {pred}, Probability: {prob[0][0]}', use_column_width=True)

            
                if prob[0][0] > 0.99:
                    pygame.mixer.music.load("alert2.mp3")
                    pygame.mixer.music.play()
    else:
        cap = cv2.VideoCapture(0)

        pygame.mixer.init()


        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Unable to access camera.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pred, prob = predict_accident(frame_rgb)

            if prob[0][0] > 0.95:
                st.image(frame_rgb, caption=f'Prediction: {pred}, Probability: {prob[0][0]}', use_column_width=True)

            if prob[0][0] > 0.99:
                pygame.mixer.music.load("alert2.mp3")
                pygame.mixer.music.play()

if __name__ == '__main__':
    main()