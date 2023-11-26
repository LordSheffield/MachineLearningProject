import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import keras
import tensorflow as tf
import uuid

#GPU Growth Limiters
com_gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in com_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Creation of Paths
POS_PATH = os.path.join('data', 'positive') 
NEG_PATH = os.path.join('data', 'negative') 
ANC_PATH = os.path.join('data', 'anchor') 


'''
#Webcam Collection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #Slice Picture into 250px by 250px
    frame = frame[120:120+250, 200:200+250, :]

    #Anchor Collection
    #image captured or passed to the program
    if cv2.waitKey(1) & 0XFF == ord('a'):
        picname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(picname, frame)

    #Positive Collection
    #image that the anchor should be and compare against it to verify the anchor.
    if cv2.waitKey(1) & 0XFF == ord('p'):
        picname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(picname, frame)

    cv2.imshow('Images Capture', frame)

    #Close Image Collection
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)

def preprocess(file_path):
    #Reading the Image
    byte_pic = tf.io.read_file(file_path)
    #Loading image
    pic = tf.io.decode_jpeg(byte_pic)
    #Prerocessing the image
    pic = tf.image.resize(pic, (100, 100))
    pic = pic / 255.0
    return pic

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

def twin_preprocess(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

data =data.map(twin_preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

def make_embedding():
    inplayer = keras.Input(shape= (100,100,3), name = "input_pic")

    #1st block
    conv1 = keras.layers.Conv2D(64,(10,10), activation = "relu")(inplayer)
    maxp1 = keras.layers.MaxPooling2D(64, (2,2), padding="same")(conv1)

    #2nd block
    conv2 = keras.layers.Conv2D(128,(7,7), activation = "relu")(maxp1)
    maxp2 = keras.layers.MaxPooling2D(64,(2,2),padding="same")(conv2)

    #3rd block
    conv3 = keras.layers.Conv2D(128,(4,4),activation="relu")(maxp2)
    maxp3 = keras.layers.MaxPooling2D(64,(2,2),padding="same")(conv3)

    #Final embed block
    conv4 = keras.layers.Conv2D(256,(4,4),activation="relu")(maxp3)
    flat1 = keras.layers.Flatten()(conv4)
    dense1 = keras.layers.Dense(4096, activation="sigmoid")(flat1)

    return keras.Model(inputs=[inplayer], outputs=[dense1] , name="embedding")


embedding = make_embedding()
embedding.summary()

class DistLayer(keras.layers.Layer):
    #Inheritance
    def __init__(self, **kwargs):
            super().__init__()
    #Similarity Calc
    def call(self,in_embed,valid_embed):
        return tf.math.abs(in_embed - valid_embed)

input_image = keras.Input(name="input_img", shape=(100,100,3))
validation_image = keras.Input(name="validation_img", shape=(100,100,3))

inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)

siamese_layer = DistLayer()
distances = siamese_layer(inp_embedding,val_embedding)
classifier = keras.layers.Dense(1,activation='sigmoid')(distances)

siamesenetwork = keras.Model(inputs=[input_image,validation_image], outputs=classifier, name= "Siamese_Network")
siamesenetwork.summary()

def make_siamese_model():
    #Anchor input
    input_image = keras.Input(name="input_image", shape=(100,100,3))
    #Validation input
    validation_image = keras.Input(name="validation_image",shape=(100,100,3))
    #combine dist components
    siamese_layer= DistLayer()
    siamese_layer._name = "distance"
    distances = siamese_layer(embedding(input_image),embedding(validation_image))

    #classification layer
    classifier = keras.layers.Dense(1,activation="sigmoid")(distances)
    
    return keras.Model(inputs=[input_image,validation_image], outputs=classifier, name= "Siamese_Network")

sia_model = make_siamese_model()
sia_model.summary()


#Loss function
binarycrossloss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4) #learning rate 0.0001

chkpt_dir = './training_checkpoints'
checkpoint_prefix =os.path.join(chkpt_dir,"chkpt")
checkpoint = tf.train.Checkpoint(opt=optimizer, sia_model=sia_model)

test_batch = train_data.as_numpy_iterator()
batch1 = test_batch.next()



@tf.function
def trainstep(batch):
    #recording operations
    with tf.GradientTape() as tape:
        #Get Anchor and pos/neg img
        x = batch[:2]
        #Get label
        yact = batch[2]

        #forward pass
        yexp = sia_model(x, training=True)
        #calc loss
        loss = binarycrossloss(yact,yexp)
    print(loss)
    #calc gradient
    grad = tape.gradient(loss,sia_model.trainable_variables)

    #calc updated wieght and apply them
    optimizer.apply_gradients(zip(grad,sia_model.trainable_variables))
    return loss

def train_loop(data, Epochs):
    for epoch in range(1, Epochs+1):
        print('\ Epoch{}/{}'.format(epoch,Epochs))
        progbar = tf.keras.utils.Progbar(len(data))
        
        #loop through each batch
        for idx, batch in enumerate(data):
            trainstep(batch)
            progbar.update(idx+1)

        #Save chkpts
        if epoch % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)


Epochs = 20
train_loop (train_data,Epochs)

#A batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
#Prediction
y_expected = sia_model.predict([test_input, test_val])

#postprocessing
res = []
for prediction in y_expected:
    if prediction > 0.5:
        res.append(1)
    else:
        res.append(0)
print(res)
#Creating recall metric
metrec = tf.keras.metrics.Recall()
#Calc recall value
metrec.update_state(y_true,y_expected)
#Return Recall Result
print("rec", metrec.result().numpy())
#precision metric
metprecision = tf.keras.metrics.Precision()
metprecision.update_state(y_true,y_expected)
print("pre",metprecision.result().numpy())

#visualizing results
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.imshow(test_input[0])
plt.subplot(1,2,2)
plt.imshow(test_val[0])
plt.show()


#Saving model
sia_model.save("siamesemodel.h5")

#Reload model
model=tf.keras.models.load_model("siamesemodel.h5",custom_objects={"DistLayer":DistLayer,
                                                                   "BinaryCrossentropy":tf.losses.BinaryCrossentropy})

model.predict([test_input,test_val])
model.summary()
