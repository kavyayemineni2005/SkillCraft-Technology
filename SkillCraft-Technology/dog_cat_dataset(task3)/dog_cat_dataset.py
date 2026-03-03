import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================
# CHANGE THIS PATH IF NEEDED
# ==========================
dataset_path = r"D:\dog_cat_dataset\assets"

X = []
y = []

print("Loading images...")

# ==========================
# LOAD IMAGES
# ==========================
for file in os.listdir(dataset_path):

    img_path = os.path.join(dataset_path, file)
    img = cv2.imread(img_path)

    if img is None:
        print("Image not loaded:", file)
        continue

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    X.append(img.flatten())

    if "cat" in file.lower():
        y.append(0)
    elif "dog" in file.lower():
        y.append(1)

X = np.array(X)
y = np.array(y)

print("Total images loaded:", len(X))
print("Classes found:", np.unique(y))

if len(np.unique(y)) < 2:
    print("Need both Cat and Dog images!")
    exit()

# ==========================
# SPLIT DATA
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ==========================
# SCALE FEATURES
# ==========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# TRAIN MODEL
# ==========================
print("Training model...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# ==========================
# TEST MODEL
# ==========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# ==========================
# SAVE PREDICTED IMAGE
# ==========================
sample_image = X_test[0]
prediction = model.predict([sample_image])[0]

sample_image_display = sample_image.reshape(64, 64).astype('uint8')

if prediction == 0:
    label = "Cat"
else:
    label = "Dog"

output_filename = "predicted_" + label + ".jpg"

cv2.imwrite(output_filename, sample_image_display)

print("Predicted image saved as:", output_filename)