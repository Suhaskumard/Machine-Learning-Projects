import os
import numpy as np
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Parameters
INIT_LR = 1e-4
EPOCHS = 10
BS = 32

print("[INFO] Generating synthetic dataset...")
NUM_SAMPLES = 1000
IMG_SIZE = 224

data = []
labels = []

def generate_face(with_mask=False):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 220  # Skin color
    
    # Face oval
    cv2.ellipse(img, (112, 100), (50, 70), 0, 0, 360, (200, 150, 100), -1)
    
    # Eyes
    cv2.circle(img, (85, 85), 8, (0,0,0), -1)
    cv2.circle(img, (139, 85), 8, (0,0,0), -1)
    
    # Mouth
    cv2.ellipse(img, (112, 140), (25, 10), 0, 0, 180, (0,0,0), 2)
    
    if with_mask:
        # Mask
        cv2.rectangle(img, (65, 95), (159, 155), (100, 150, 255), -1)
        cv2.rectangle(img, (65, 95), (159, 155), (50, 50, 200), 3)
    
    return img.astype(np.float32) / 255.0

for i in range(NUM_SAMPLES):
    img = generate_face(with_mask=True)
    data.append(img)
    labels.append('with_mask')
    
    img = generate_face(with_mask=False)
    data.append(img)
    labels.append('without_mask')

data = np.array(data)
labels = np.array(labels)
print(f"[INFO] Generated {len(data)} synthetic images")

# Encode labels
lb = LabelBinarizer()
labels_bin = lb.fit_transform(labels)
labels = np.hstack((labels_bin, 1 - labels_bin))

# Train-Test split
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# Load base model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# Build head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] Training...")
model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BS,
          validation_data=(testX, testY))

# Save model
model.save("mask_model.h5")
print("[INFO] Model saved!")
