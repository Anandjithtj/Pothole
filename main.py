import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.utils import shuffle

# Set the input shape and image size
size = 100

# Define the Keras model
def kerasModel4():
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(size, size, 3)))  # 3 channels for RGB
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(2))  # Output layer for two classes (Pothole, Plain)
    model.add(Activation('softmax'))
    return model

# Check if directories exist
print(f"Does 'images/pothole' exist? {os.path.exists('images/pothole')}")
print(f"Does 'images/plain' exist? {os.path.exists('images/plain')}")

# Load Training data: pothole
potholeTrainImages = glob.glob("E:/pothole anto code/potooo/images/pothole/train/*.jpg")
potholeTrainImages.extend(glob.glob("E:/pothole anto code/potooo/images/pothole/train/*.jpeg"))
potholeTrainImages.extend(glob.glob("E:/pothole anto code/potooo/images/pothole/train/*.png"))
print(f"Found {len(potholeTrainImages)} pothole training images: {potholeTrainImages}")

train1 = []
for img in potholeTrainImages:
    image = cv2.imread(img)  # Load as RGB
    if image is not None:
        train1.append(image)
    else:
        print(f"Failed to load image: {img}")

# Load Training data: non-pothole
nonPotholeTrainImages = glob.glob("E:/pothole anto code/potooo/images/plain/train/*.jpg")
nonPotholeTrainImages.extend(glob.glob("E:/pothole anto code/potooo/images/plain/train/*.jpeg"))
nonPotholeTrainImages.extend(glob.glob("E:/pothole anto code/potooo/images/plain/train/*.png"))
print(f"Found {len(nonPotholeTrainImages)} non-pothole training images: {nonPotholeTrainImages}")

train2 = []
for img in nonPotholeTrainImages:
    image = cv2.imread(img)  # Load as RGB
    if image is not None:
        train2.append(image)
    else:
        print(f"Failed to load image: {img}")

# Resize images
for i in range(len(train1)):
    train1[i] = cv2.resize(train1[i], (size, size))

for i in range(len(train2)):
    train2[i] = cv2.resize(train2[i], (size, size))

# Convert to numpy arrays
temp1 = np.asarray(train1)
temp2 = np.asarray(train2)

# Load Testing data: potholes
potholeTestImages = glob.glob("E:/pothole anto code/potooo/images/pothole/test/*.jpg")
test1 = []
for img in potholeTestImages:
    image = cv2.imread(img)  # Load as RGB
    if image is not None:
        test1.append(image)
    else:
        print(f"Failed to load image: {img}")
for i in range(len(test1)):
    test1[i] = cv2.resize(test1[i], (size, size))
temp3 = np.asarray(test1)

# Load Testing data: non-potholes
nonPotholeTestImages = glob.glob("E:/pothole anto code/potooo/images/plain/test/*.jpg")
nonPotholeTestImages.extend(glob.glob("E:/pothole anto code/potooo/images/plain/test/*.png"))
nonPotholeTestImages.extend(glob.glob("E:/pothole anto code/potooo/images/plain/test/*.jpeg"))
test2 = []
for img in nonPotholeTestImages:
    image = cv2.imread(img)  # Load as RGB
    if image is not None:
        test2.append(image)
    else:
        print(f"Failed to load image: {img}")
for i in range(len(test2)):
    test2[i] = cv2.resize(test2[i], (size, size))
temp4 = np.asarray(test2)

# Prepare the training and testing datasets
X_train = []
X_train.extend(temp1)
X_train.extend(temp2)
X_train = np.asarray(X_train)

X_test = []
X_test.extend(temp3)
X_test.extend(temp4)
X_test = np.asarray(X_test)

# Check if any data is empty
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Create labels for the data
y_train1 = np.ones([temp1.shape[0]], dtype=int)
y_train2 = np.zeros([temp2.shape[0]], dtype=int)
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)

y_train = []
y_train.extend(y_train1)
y_train.extend(y_train2)
y_train = np.asarray(y_train)

y_test = []
y_test.extend(y_test1)
y_test.extend(y_test2)
y_test = np.asarray(y_test)

# Check if labels are correctly generated
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Shuffle the data
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# Normalize the data
X_train = X_train / 255.0  # Normalize to [0, 1]
X_test = X_test / 255.0

# Convert labels to categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Check final shapes
print(f"X_train shape after normalization: {X_train.shape}")
print(f"X_test shape after normalization: {X_test.shape}")

# Define the model
inputShape = (size, size, 3)  # 3 channels for RGB
model = kerasModel4()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=500, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print(f'{metric_name}: {metric_value}')

# Save the model and weights
print("Saving model and weights...")
model.save('pothole_model.keras')  # Recommended format
model.save_weights('pothole_weights.weights.h5')  # Correct weights format

# Save the model architecture (optional)
model_json = model.to_json()
with open("pothole_model.json", "w") as json_file:
    json_file.write(model_json)

print("Model and weights saved successfully.")