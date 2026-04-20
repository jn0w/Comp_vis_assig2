from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Rescaling,
    BatchNormalization, RandomFlip, RandomRotation, RandomZoom,
    RandomTranslation,
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
num_classes = 3
img_width = 224
img_height = 224
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
print('Input size: {}x{}, batch_size={}'.format(img_height, img_width, batch_size))
print('Class Names: ', class_names)
num_classes = len(class_names)

print('Training samples per class: BACTERIAL=2596, NORMAL=1461, VIRAL=1362')

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

if fit:
    # Increased resolution to 224x224 — MobileNetV2's native ImageNet size.
    # More spatial detail helps distinguish bacterial vs viral patterns.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, img_channels),
        include_top=False,
        weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.06),
        RandomZoom(height_factor=(-0.06, 0.06), width_factor=(-0.06, 0.06)),
        RandomTranslation(0.04, 0.04),
        Rescaling(1.0 / 127.5, offset=-1.0),  # MobileNetV2 expects [-1, 1]
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'])

    model.summary()

    print('\n=== Phase 1: Training classifier head ===')
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        "pneumonia.keras", monitor='val_accuracy',
        save_freq='epoch', save_best_only=True)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[save_callback],
        epochs=10)

    print('\n=== Phase 2: Fine-tuning top layers ===')
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=5e-5),
                  metrics=['accuracy'])

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[save_callback, earlystop_callback],
        epochs=15)

    history_acc = history1.history['accuracy'] + history2.history['accuracy']
    history_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    history_loss = history1.history['loss'] + history2.history['loss']
    history_val_loss = history1.history['val_loss'] + history2.history['val_loss']
else:
    model = tf.keras.models.load_model("pneumonia.keras")

score = model.evaluate(test_ds, batch_size=batch_size)
print('Test accuracy:', score[1])

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
    ax1.plot(history_acc)
    ax1.plot(history_val_acc)
    ax1.axvline(x=len(history1.history['accuracy']) - 0.5,
                color='gray', linestyle='--', label='Fine-tune start')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val', 'Fine-tune start'], loc='upper left')

    ax2.plot(history_loss)
    ax2.plot(history_val_loss)
    ax2.axvline(x=len(history1.history['loss']) - 0.5,
                color='gray', linestyle='--', label='Fine-tune start')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val', 'Fine-tune start'], loc='upper right')
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
