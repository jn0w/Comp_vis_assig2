from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D,
    Rescaling, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
num_classes = 3
epochs = 20
img_width = 128
img_height = 128
img_channels = 3
fit = True

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(_SCRIPT_DIR, 'dataset', 'chest_xray', 'train')
test_dir = os.path.join(_SCRIPT_DIR, 'dataset', 'chest_xray', 'test')

_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpus[0], True)
    except (ValueError, RuntimeError):
        pass

train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    validation_split=0.2,
    subset='both',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=True)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=False)

class_names = train_ds.class_names
print('Class Names: ', class_names)
num_classes = len(class_names)

# Class weights to handle imbalanced dataset
class_counts = np.array([2596, 1461, 1362])  # BACTERIAL, NORMAL, VIRAL
total = class_counts.sum()
class_weight = {i: total / (num_classes * c) for i, c in enumerate(class_counts)}
print('Class weights:', class_weight)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(2):
    for i in range(min(6, len(images))):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.show()

# Deeper CNN with augmentation and regularisation
model = Sequential([
    Rescaling(1.0 / 255),

    RandomFlip("horizontal"),
    RandomRotation(0.05),
    RandomZoom(0.05),

    Conv2D(32, (3, 3), activation='relu', padding='same',
           input_shape=(img_height, img_width, img_channels)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=2e-4),
              metrics=['accuracy'])

model.summary()

earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
save_callback = tf.keras.callbacks.ModelCheckpoint(
    "pneumonia.keras", save_freq='epoch', save_best_only=True)

if fit:
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[save_callback, earlystop_callback, reduce_lr_callback],
        epochs=epochs)
else:
    model = tf.keras.models.load_model("pneumonia.keras")

score = model.evaluate(test_ds, batch_size=batch_size)
print('Test accuracy:', score[1])

# Per-class precision, recall, f1
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print('\n===== Per-Class Results =====')
for i, name in enumerate(class_names):
    tp = np.sum((y_pred == i) & (y_true == i))
    fp = np.sum((y_pred == i) & (y_true != i))
    fn = np.sum((y_pred != i) & (y_true == i))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f'  {name:12s}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}')

overall_acc = np.sum(y_pred == y_true) / len(y_true)
print(f'\nOverall Accuracy: {overall_acc:.4f}')

if fit:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val'], loc='upper right')
    plt.tight_layout()
    plt.show()

test_batch = test_ds.take(1)
plt.figure(figsize=(10, 10))
for images, labels in test_batch:
    for i in range(min(6, len(images))):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction = model.predict(tf.expand_dims(images[i].numpy(), 0))
        plt.title('Actual:' + class_names[labels[i].numpy()] +
                  '\nPredicted:{} {:.2f}%'.format(
                      class_names[np.argmax(prediction)],
                      100 * np.max(prediction)))
        plt.axis("off")
plt.show()
