import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 🌾 Step 1: Set dataset path
base_dir = "dataset"

# 🌿 Step 2: Load and augment images
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("✅ Data loaded successfully!")

# 🌱 Step 3: Define a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 🧠 Step 4: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🚀 Step 5: Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    batch_size=32
)

# 📊 Step 6: Evaluate performance
loss, accuracy = model.evaluate(val_data)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

# 💾 Step 7: Save the model
model.save("agrimind_model.h5")
print("💾 Model saved as agrimind_model.h5")
