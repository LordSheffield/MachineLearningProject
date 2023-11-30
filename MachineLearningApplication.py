import cv2
import os
import numpy as np
import keras
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Prerocessing the image
def preprocess(file_path):
    byte_pic = tf.io.read_file(file_path)
    pic = tf.io.decode_jpeg(byte_pic)
    pic = tf.image.resize(pic, (100, 100))
    pic = pic / 255.0
    return pic

class DistLayer(keras.layers.Layer):
    #Inheritance
    def __init__(self, **kwargs):
            super().__init__()
    #Similarity Calc
    def call(self,in_embed,valid_embed):
        return tf.math.abs(in_embed - valid_embed)
    
model=tf.keras.models.load_model("siamesemodel.h5", custom_objects={"DistLayer":DistLayer, "BinaryCrossentropy":tf.losses.BinaryCrossentropy})

#Compares webcam photo with photos of Connor
def verifyCB(frame, model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'CB_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'CB_images', image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'CB_images')))
    verified = verification > verification_threshold
    if (verification > verification_threshold):
        print("Verified, user is Connor Brooks.")
    else:
        print("Unverified, user is not Connor Brooks.")

    return results, verified

#Compares webcam photo with photos of Leif
def verifyLO(frame, model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'LO_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'LO_images', image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'LO_images')))
    verified = verification > verification_threshold
    if (verification > verification_threshold):
        print("Verified, user is Leif Orth.")
    else:
        print("Unverified, user is not Leif Orth.")

    return results, verified

#Compares webcam photo with a photo of Dr. Rahimi
def verifyDocR(frame, model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'DocR_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'DocR_images', image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'DocR_images')))
    verified = verification > verification_threshold
    if (verification > verification_threshold):
        print("Verified, user is Dr. Rahimi.")
    else:
        print("Unverified, user is not Dr. Rahimi.")

    return results, verified

menu = True
user_input = 0
while menu == True:
    print("Welcome to Facial Recognition!")
    print("To Validate Connor Brooks enter 1.")
    print("To Validate Leif Orth enter 2.")
    print("To Validate Dr. Rahimi enter 3. [Experimental]")
    print("To exit application enter 4.")
    user_input = input()
    if user_input == '1':
        print("A webcam instance is at your taskbar. Center your face in frame then press 'v to validate your identity. Press 'q' to exit.")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = frame[120:120+250, 200:200+250, :]
            cv2.imshow('Facial_Verification', frame)

            if cv2.waitKey(10) & 0xFF == ord('v'):
                cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
                results, verified = verifyCB(frame, model, 0.7, 0.7)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    if user_input == '2':
        print("A webcam instance is at your taskbar. Center your face in frame then press 'v to validate your identity. Press 'q' to exit.")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = frame[120:120+250, 200:200+250, :]
            cv2.imshow('Facial_Verification', frame)

            if cv2.waitKey(10) & 0xFF == ord('v'):
                cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
                results, verified = verifyLO(frame, model, 0.7, 0.7)
                print(verified)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    if user_input == '3':
        print("A webcam instance is at your taskbar. Center your face in frame then press 'v to validate your identity. Press 'q' to exit.")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = frame[120:120+250, 200:200+250, :]
            cv2.imshow('Facial_Verification', frame)

            if cv2.waitKey(10) & 0xFF == ord('v'):
                cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
                results, verified = verifyDocR(frame, model, 0.5, 0.5)
                print(verified)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    if user_input == '4':
        menu = False
    
    else:
        print("Error! That command does not exist, please try again.")