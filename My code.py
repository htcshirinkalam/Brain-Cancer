import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set the paths to your dataset
data_dir = r'C:\Users\Asus\Desktop\Brain-Cancer\Training'  # Replace with your dataset path
categories = ['Tumor', 'No_Tumor']  # Ensure these folder names match your dataset

# Load the images and labels
def load_data(data_dir, categories):
    img_data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)  # Assign a numeric label (0 or 1)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                img_array = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
                img_array = tf.keras.preprocessing.image.img_to_array(img_array)
                img_data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(img_data), np.array(labels)

# Load dataset
X, y = load_data(data_dir, categories)

# Normalize images
X = X / 255.0  # Scale pixel values to be between 0 and 1

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the CNN model
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class problems
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create model
input_shape = (128, 128, 3)
model = create_model(input_shape)

# Train the model using the augmented images and save training history
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))


# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels
# Generate classification report and confusion matrix
print(classification_report(y_test, y_pred_classes))
confusion_mtx = confusion_matrix(y_test, y_pred_classes)
# Plot accuracy vs epochs
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_epoch.png')  # Save the plot as a file
plt.show()

