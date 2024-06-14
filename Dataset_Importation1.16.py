import os
import cv2

# Define paths
input_folder = 'yolo/obj_train_data'  # Change this to your input folder path
output_folder = os.path.join(input_folder, 'new_output')
obj_names_path = 'yolo/obj.names'  # Change this to your obj_names file path

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to read class labels from obj_names file
def read_class_labels(obj_names_path):
    with open(obj_names_path, 'r') as file:
        class_labels = file.read().splitlines()
    # Exchange class labels to meet the requirement
    class_labels[0], class_labels[1] = class_labels[1], class_labels[0]
    return class_labels

# Function to draw bounding boxes and labels on the image
def draw_bounding_boxes(image_path, annotation_path, class_labels):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
            class_id = int(class_id)
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            x_max = int(x_center + (box_width / 2))
            y_max = int(y_center + (box_width / 2))

            # Draw the rectangle on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Put class label text on the image
            label = class_labels[class_id]
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# Read class labels from obj_names file
class_labels = read_class_labels(obj_names_path)

# Process each image and its corresponding annotation
for filename in os.listdir(input_folder):
    if filename.endswith('.PNG'):
        image_path = os.path.join(input_folder, filename)
        annotation_path = os.path.join(input_folder, filename.replace('.PNG', '.txt'))

        if os.path.exists(annotation_path):
            annotated_image = draw_bounding_boxes(image_path, annotation_path, class_labels)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, annotated_image)
            print(f'Saved annotated image: {output_path}')

print('Processing complete.')

# Creating separate folders for each class
saloon_cars_folder = os.path.join(output_folder, 'saloon_cars')
motorcycles_folder = os.path.join(output_folder, 'motorcycles')

if not os.path.exists(saloon_cars_folder):
    os.makedirs(saloon_cars_folder)

if not os.path.exists(motorcycles_folder):
    os.makedirs(motorcycles_folder)

# Cropping and saving image patches
def crop_and_save(image, class_id, x_min, y_min, x_max, y_max, filename, count):
    cropped_image = image[y_min:y_max, x_min:x_max]
    if class_id == 0:
        output_path = os.path.join(saloon_cars_folder, f'saloon_cars_{count}_{filename}')
    elif class_id == 1:
        output_path = os.path.join(motorcycles_folder, f'motorcycles_{count}_{filename}')
    cv2.imwrite(output_path, cropped_image)
    print(f'Saved cropped image: {output_path}')

# Initializing a counter for each class to ensure unique filenames
saloon_cars_count = 0
motorcycles_count = 0

# Processing each image and its corresponding annotation to crop and save patches
for filename in os.listdir(input_folder):
    if filename.endswith('.PNG'):
        image_path = os.path.join(input_folder, filename)
        annotation_path = os.path.join(input_folder, filename.replace('.PNG', '.txt'))

        if os.path.exists(annotation_path):
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            with open(annotation_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                    class_id = int(class_id)
                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height

                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    x_max = int(x_center + (box_width / 2))
                    y_max = int(y_center + (box_width / 2))

                    # Cropping and saving the image patch
                    if class_id == 0:
                        saloon_cars_count += 1
                        crop_and_save(image, class_id, x_min, y_min, x_max, y_max, filename, saloon_cars_count)
                    elif class_id == 1:
                        motorcycles_count += 1
                        crop_and_save(image, class_id, x_min, y_min, x_max, y_max, filename, motorcycles_count)

print('Cropping and saving complete.')


# Loading the dataset pipeline using kera library
# Images were transfered manually from the folder new_output folder to newly created folder, Datase_cropped
import tensorflow as tf

dataset_new = tf.keras.utils.image_dataset_from_directory('yolo/obj_train_data/new_output/Dataset_cropped',batch_size = 32,image_size = (224,224))

scaled_dataset =dataset_new.map(lambda x,y: (x/255,y))

scaled_dataset.as_numpy_iterator().next()[0]

# Declaring the train and test size of the dataset
train_size = int(0.8 * len(scaled_dataset))
test_size = int(0.2 * len(scaled_dataset))

train_dataset = scaled_dataset.take(train_size) # Extract training dataset
test_dataset = scaled_dataset.skip(train_size).take(test_size) # Extract test dataset

# CNN MODEL

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import matplotlib.pyplot as plt

# Defining the model
CNN_model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling the model
optimizer = Adam(learning_rate=0.001)
CNN_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
hist = CNN_model.fit(train_dataset, validation_data=test_dataset, epochs=2, batch_size=32, validation_split=0.2)

# Evalauting the model
from tensorflow.keras.metrics import Precision,Recall,BinaryAccuracy

Pre = Precision()
Rec = Recall()
Acc=BinaryAccuracy()

for batch in test_dataset.as_numpy_iterator():
    X, y = batch
    y_pred = CNN_model.predict(X)
    Pre.update_state(y,y_pred)
    Rec.update_state(y,y_pred)
    Acc.update_state(y,y_pred)
    print(f'Precision:{Pre.result().numpy()}')
    print(f'Recall:{Rec.result().numpy()}')
    print(f'Accuracy:{Acc.result().numpy()}')

# Plotting performance

plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='blue', label='val_accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# ResNet MODEL

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Creating a residual block
def residual_block(x, filters, kernel_size=3, stride=1, activation='relu'):
    y = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)
    
    y = Conv2D(filters, kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    
    if stride != 1 or x.shape[-1] != filters:
        x = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
    
    out = Add()([x, y])
    out = Activation(activation)(out)
    return out

# Defining input layer
input_layer = Input(shape=(224, 224, 3))

# Building initial convolutional layer
x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# Adding residual blocks
x = residual_block(x, 64)
x = residual_block(x, 64)

x = residual_block(x, 128, stride=2)
x = residual_block(x, 128)

x = residual_block(x, 256, stride=2)
x = residual_block(x, 256)

x = residual_block(x, 512, stride=2)
x = residual_block(x, 512)

# Defining final layers
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)

# Creating the model
ResNet_model = Model(inputs=input_layer, outputs=output_layer)

# Compiling the model
ResNet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Printing the model summary
ResNet_model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Setting up callbacks
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training the model
history1 = ResNet_model.fit(train_dataset, epochs=2, validation_data=test_dataset, callbacks=[checkpoint, early_stopping])


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# Defining metrics
Pre = Precision()
Rec = Recall()
Acc = BinaryAccuracy()

# Using the test_set
for batch in test_dataset.as_numpy_iterator():
    X, y = batch
    
    y_pred = ResNet_model.predict(X)
    
    # Updating metrics
    Pre.update_state(y, y_pred)
    Rec.update_state(y, y_pred)
    Acc.update_state(y, y_pred)
    
    # Printing current metric values
    print(f'Precision: {Pre.result().numpy()}')
    print(f'Recall: {Rec.result().numpy()}')
    print(f'Accuracy: {Acc.result().numpy()}')

# Plotting performance of the model
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(history1.history['accuracy'], color='teal', label='accuracy')
plt.plot(history1.history['val_accuracy'], color='blue', label='val_accuracy')
fig.suptitle('ResNet MODEL ACCURACY', fontsize=20)
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# GoogLeNet MODEL (Also known as Inception v1 model)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, concatenate, Input

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=-1)
    return output

input_layer = Input(shape=(224, 224, 3))

x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, 64, 96, 128, 16, 32, 32)
x = inception_module(x, 128, 128, 192, 32, 96, 64)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, 192, 96, 208, 16, 48, 64)
x = inception_module(x, 160, 112, 224, 24, 64, 64)
x = inception_module(x, 128, 128, 256, 24, 64, 64)
x = inception_module(x, 112, 144, 288, 32, 64, 64)
x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = inception_module(x, 384, 192, 384, 48, 128, 128)

x = AveragePooling2D((7, 7), strides=(1, 1))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
output_layer = Dense(1, activation='sigmoid')(x)

googlenet_model = Model(inputs=input_layer, outputs=output_layer)

googlenet_model.summary()

#install scikit-learn from command prompt
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Compile the model
googlenet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
googlenet_model.fit(train_dataset, epochs=2)

# Evaluate the model
predictions = googlenet_model.predict(test_dataset)
predictions = np.round(predictions).flatten()  # Converting probabilities to binary predictions

# Calculating accuracy
accuracy = accuracy_score(test_labels, predictions)

# Calculating precision and recall
precision = Precision()
recall = Recall()
precision.update_state(test_labels, predictions)
recall.update_state(test_labels, predictions)
precision_result = precision.result().numpy()
recall_result = recall.result().numpy()

# Calculating F1 score
f1 = 2 * (precision_result * recall_result) / (precision_result + recall_result)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision_result}")
print(f"Recall: {recall_result}")
print(f"F1 Score: {f1}")


# YOLOv4 MODEL

# YOLOv4 model definition
def conv_block(x, filters, kernel_size, strides=1, padding='same'):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def darknet_body(x):
    x = conv_block(x, 32, 3)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = conv_block(x, 64, 3)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = conv_block(x, 128, 3)
    x = conv_block(x, 64, 1)
    x = conv_block(x, 128, 3)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = conv_block(x, 256, 3)
    x = conv_block(x, 128, 1)
    x = conv_block(x, 256, 3)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = conv_block(x, 512, 3)
    for _ in range(3):
        x = conv_block(x, 256, 1)
        x = conv_block(x, 512, 3)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    for _ in range(2):
        x = conv_block(x, 1024, 3)
    return x

def yolo_v4_model():
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))  # Update the input shape to match your dataset
    x = darknet_body(input_layer)

    # Detection Head
    x = conv_block(x, 512, 1)
    x = conv_block(x, 1024, 3)
    x = conv_block(x, 512, 1)
    x = conv_block(x, 1024, 3)
    x = conv_block(x, 512, 1)

    # Output
    output = tf.keras.layers.Conv2D(255, 1, strides=1, padding='same')(x)  # Assuming 255 is the number of filters for prediction
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    return model

# Custom loss and accuracy
def yolo_loss(y_true, y_pred):
    # Implementing YOLOv4 loss function here
    return tf.reduce_mean(tf.square(y_true - y_pred))

def custom_accuracy(y_true, y_pred):
    # Implementing custom accuracy calculation here
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(y_pred)), dtype=tf.float32))

# Creating YOLOv4 model
yolov4_model = yolo_v4_model()
yolov4_model.compile(optimizer='adam', loss=yolo_loss, metrics=[custom_accuracy])

# Training YOLOv4 model
history4 = yolov4_model.fit(train_dataset, validation_data=test_dataset, epochs=10)

# Evaluating YOLOv4 model
test_loss, test_accuracy = yolov4_model.evaluate(test_dataset)
print("Test Accuracy:", test_accuracy)

test_predictions = yolov4_model.predict(test_dataset)

# Converting predictions to class labels (you need to define this conversion logic based on your model output)
def convert_predictions_to_classes(predictions):
    # Implement your logic to convert predictions to class labels
    # For this example, we assume a simple argmax
    return np.argmax(predictions, axis=-1)

test_pred_classes = convert_predictions_to_classes(test_predictions)

# Extracting labels from test set
test_labels = []
for images, labels in test_dataset:
    test_labels.extend(labels.numpy())
test_labels = np.array(test_labels)

# Computing metrics
test_precision = precision_score(test_labels, test_pred_classes, average='weighted')
test_recall = recall_score(test_labels, test_pred_classes, average='weighted')
test_f1 = f1_score(test_labels, test_pred_classes, average='weighted')

print("\nTest Set Metrics:")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")

# Plotting performance graph
train_loss = history4.history['loss']
val_loss = history4.history['val_loss']
train_accuracy = history.history['custom_accuracy']
val_accuracy = history.history['val_custom_accuracy']

# Plot loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
