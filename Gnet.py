
# Loading the dataset pipeline using kera library
# Images were transfered manually from the folder new_output folder to newly created folder, Datase_cropped
import tensorflow as tf
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt


dataset_new = tf.keras.utils.image_dataset_from_directory('yolo/obj_train_data/new_output/Dataset_cropped',batch_size = 32,image_size = (224,224))

scaled_dataset =dataset_new.map(lambda x,y: (x/255,y))

scaled_dataset.as_numpy_iterator().next()[0]

# Declaring the train and test size of the dataset
train_size = int(0.8 * len(scaled_dataset))
test_size = int(0.2 * len(scaled_dataset))

train_dataset = scaled_dataset.take(train_size) # Extract training dataset
test_dataset = scaled_dataset.skip(train_size).take(test_size) # Extract test dataset


# BUILDING A GOOGLENET CONVOLUTIONAL NEURAL NETWORK
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, concatenate
from tensorflow.keras.models import Model

# Function to create an inception module
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

# Define the input layer
input_layer = Input(shape=(224, 224, 3))

# Initial layers before Inception modules
x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# Add inception modules
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

# Final layers
x = AveragePooling2D((7, 7), strides=(1, 1))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
output_layer = Dense(1, activation='sigmoid')(x)

# Create the model
googlenetmodel = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
googlenetmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary

# Define number of epochs and batch size
epochs = 2
batch_size = 32

# Train the model
history = googlenetmodel.fit(train_dataset,
                             epochs=epochs,
                             batch_size=batch_size,
                             validation_data=test_dataset)

googlenetmodel.summary()

# Evaluate the model
from tensorflow.keras.metrics import Precision,Recall,BinaryAccuracy

Pre = Precision()
Rec = Recall()
Acc=BinaryAccuracy()
for batch in test_dataset.as_numpy_iterator():
    X, y = batch
    y_pred = googlenetmodel.predict(X)
    # Updating metrics
    Pre.update_state(y, y_pred)
    Rec.update_state(y, y_pred)
    Acc.update_state(y, y_pred)
    
    # Printing current metric values
    print(f'Precision: {Pre.result().numpy()}')
    print(f'Recall: {Rec.result().numpy()}')
    print(f'Accuracy: {Acc.result().numpy()}')


# plot performance
fig = plt.figure()
plt.plot(history.history['accuracy'],color = 'teal',label = 'accuracy')
plt.plot(history.history['val_accuracy'],color = 'blue',label = 'val_accuracy')
plt.title('GoogleNet Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
