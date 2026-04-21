from __future__ import print_function

import os

# Deep learning library (TensorFlow/Keras) + helper layers used in the model.
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt  # used to show images and training graphs
import numpy as np               # used for simple math on arrays

# ---- Basic settings ----
batch_size = 32          # how many images the model looks at per step
num_classes = 3          # we are predicting 3 classes: BACTERIAL, NORMAL, VIRAL
# Images are resized to 224x224 because EfficientNetB0 was trained on that size.
# If your GPU runs out of memory, lower batch_size to 16.
img_width = 224
img_height = 224
img_channels = 3         # 3 colour channels (RGB)
fit = True               # True = train a new model, False = load saved model

# ---- Dataset folder paths (located next to this script) ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(_SCRIPT_DIR, 'dataset', 'chest_xray', 'train')
test_dir = os.path.join(_SCRIPT_DIR, 'dataset', 'chest_xray', 'test')

# Let the GPU grow its memory use as needed (avoids grabbing all memory at once).
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpus[0], True)
    except (ValueError, RuntimeError):
        pass

# ---- Load training data ----
# Read images from folders and split 80% for training, 20% for validation.
# Class labels are taken automatically from the subfolder names.
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    validation_split=0.2,
    subset='both',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=True)

# ---- Load test data (separate folder, never shuffled so results are consistent) ----
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=False)

# Grab the class name list (e.g. ['BACTERIAL', 'NORMAL', 'VIRAL']).
class_names = train_ds.class_names
print('Input size: {}x{}, batch_size={}'.format(img_height, img_width, batch_size))
print('Class Names: ', class_names)
num_classes = len(class_names)

# ---- Class weights (handle imbalanced dataset) ----
# There are many more BACTERIAL images than NORMAL/VIRAL, so the model might
# ignore the smaller classes. We give the smaller classes a slightly higher
# weight during training so they still get attention.
# The sqrt() makes this adjustment softer (less aggressive than full balancing).
_counts_by_name = {'BACTERIAL': 2596, 'NORMAL': 1461, 'VIRAL': 1362}
_counts = np.array([_counts_by_name[n] for n in class_names], dtype=np.float64)
_k = float(len(class_names))
_balanced = _counts.sum() / (_k * _counts)   # standard balanced weights
_soft = np.sqrt(_balanced)                   # soften the effect
_soft *= _k / _soft.sum()                    # rescale so weights average to 1
class_weight = {i: float(_soft[i]) for i in range(len(class_names))}
print('Soft sqrt class weights (help minority class, milder than full balanced):', class_weight)

# ---- Speed up training by caching images in memory and preloading batches ----
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---- Show a few training images so we can visually check the data ----
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(2):
    for i in range(min(6, len(images))):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.show()

if fit:
    # =====================================================================
    # FINAL MODEL - Transfer learning with EfficientNetB0
    # ---------------------------------------------------------------------
    # We don't train a CNN from scratch. Instead we reuse EfficientNetB0,
    # a strong image model already trained on millions of photos (ImageNet).
    # We keep its learned "eye for features" and only teach it our 3 classes.
    # EfficientNetB0 already rescales/normalises inputs internally,
    # so we can pass images in the normal [0, 255] RGB range.
    # =====================================================================
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_height, img_width, img_channels),
        include_top=False,      # drop the original 1000-class classifier
        weights='imagenet')     # load the pretrained weights
    base_model.trainable = False  # freeze the backbone for Phase 1

    # ---- Build the full model as a stack of layers ----
    # Step 1: small random image tweaks (rotation, zoom, shift) so the model
    #         learns to handle real-world variations (data augmentation).
    # Step 2: EfficientNet backbone turns the image into useful features.
    # Step 3: GlobalAveragePooling turns the feature maps into one vector.
    # Step 4: Two Dense layers learn patterns specific to pneumonia classes.
    #         BatchNormalization helps training stay stable.
    #         Dropout randomly "switches off" neurons to reduce overfitting.
    # Step 5: Final Dense(softmax) outputs a probability for each class.
    model = tf.keras.Sequential([
        RandomRotation(0.06),
        RandomZoom(height_factor=(-0.06, 0.06), width_factor=(-0.06, 0.06)),
        RandomTranslation(0.04, 0.04),
        base_model,
        GlobalAveragePooling2D(),
        Dense(320, activation='relu'),
        BatchNormalization(),
        Dropout(0.28),
        Dense(160, activation='relu'),
        BatchNormalization(),
        Dropout(0.22),
        Dense(num_classes, activation='softmax')
    ])

    # Compile = choose how the model learns:
    # - loss: measures how wrong the predictions are (for integer labels).
    # - optimizer: Adam adjusts the weights; lr=1e-3 is a normal starting speed.
    # - metric: track accuracy during training.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'])

    model.summary()  # print the layer-by-layer summary of the model

    # =====================================================================
    # PHASE 1 - Train only the new classifier head
    # ---------------------------------------------------------------------
    # The EfficientNet backbone is frozen, so only the new Dense layers on
    # top learn. This is fast and safely adapts the model to our 3 classes
    # without damaging the useful features already inside EfficientNet.
    # =====================================================================
    print('\n=== Phase 1: Training classifier head ===')
    # Save the best version of the model (highest validation accuracy) to disk.
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        "pneumonia.keras", monitor='val_accuracy',
        save_freq='epoch', save_best_only=True)

    # Train for 12 epochs, using class_weight to balance class sizes.
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[save_callback],
        epochs=12)

    # =====================================================================
    # PHASE 2 - Fine-tune the top of EfficientNet
    # ---------------------------------------------------------------------
    # Now we unfreeze the last ~55 layers of the backbone and continue
    # training with a very small learning rate (4e-5). This lets the model
    # slightly adjust its high-level features for chest X-rays, without
    # forgetting everything it already learned from ImageNet.
    # =====================================================================
    print('\n=== Phase 2: Fine-tuning top layers ===')
    base_model.trainable = True
    fine_tune_at = 55  # number of top layers to unfreeze
    for layer in base_model.layers[:-fine_tune_at]:
        layer.trainable = False  # keep the lower (earlier) layers frozen

    # Re-compile with a much smaller learning rate for gentle fine-tuning.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=4e-5),
                  metrics=['accuracy'])

    # Extra callbacks for Phase 2:
    # - EarlyStopping: stop training if val accuracy stops improving for 9 epochs
    #   (and restore the best weights seen so far).
    # - ReduceLROnPlateau: halve the learning rate if val loss plateaus for 4 epochs.
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=9, restore_best_weights=True)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7)

    # Train for up to 28 more epochs (EarlyStopping may end it sooner).
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[save_callback, earlystop_callback, reduce_lr_callback],
        epochs=28)

    # Join the Phase 1 and Phase 2 histories so we can plot one continuous graph.
    history_acc = history1.history['accuracy'] + history2.history['accuracy']
    history_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    history_loss = history1.history['loss'] + history2.history['loss']
    history_val_loss = history1.history['val_loss'] + history2.history['val_loss']
else:
    # If fit=False, skip training and load the previously saved best model.
    model = tf.keras.models.load_model("pneumonia.keras")

# ---- Evaluate the model on the unseen test set ----
# This gives us the final accuracy we report.
score = model.evaluate(test_ds, batch_size=batch_size)
print('Test accuracy:', score[1])

# ---- Per-class Precision, Recall and F1-score ----
# Accuracy alone isn't enough with an unbalanced dataset, so we also
# calculate how well the model does on EACH class individually.
#   Precision = of all predictions for this class, how many were correct?
#   Recall    = of all real images of this class, how many did we catch?
#   F1        = balanced average of precision and recall
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])  # real labels
y_pred_probs = model.predict(test_ds)                                # class probs
y_pred = np.argmax(y_pred_probs, axis=1)                             # picked class

print('\n===== Per-Class Results =====')
for i, name in enumerate(class_names):
    # tp = correctly predicted as this class
    # fp = predicted as this class but actually another class
    # fn = actually this class but predicted as another class
    tp = np.sum((y_pred == i) & (y_true == i))
    fp = np.sum((y_pred == i) & (y_true != i))
    fn = np.sum((y_pred != i) & (y_true == i))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f'  {name:12s}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}')

# Overall accuracy computed manually as a sanity check.
overall_acc = np.sum(y_pred == y_true) / len(y_true)
print(f'\nOverall Accuracy: {overall_acc:.4f}')

# ---- Plot accuracy and loss curves for the whole training run ----
# Left graph shows accuracy per epoch, right graph shows loss per epoch.
# A dashed vertical line marks where Phase 2 (fine-tuning) started.
if fit:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history_acc)       # training accuracy
    ax1.plot(history_val_acc)   # validation accuracy
    ax1.axvline(x=len(history1.history['accuracy']) - 0.5,
                color='gray', linestyle='--', label='Fine-tune start')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val', 'Fine-tune start'], loc='upper left')

    ax2.plot(history_loss)      # training loss
    ax2.plot(history_val_loss)  # validation loss
    ax2.axvline(x=len(history1.history['loss']) - 0.5,
                color='gray', linestyle='--', label='Fine-tune start')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val', 'Fine-tune start'], loc='upper right')
    plt.tight_layout()
    plt.show()

# ---- Visual sanity check: show 6 test images with predicted vs actual class ----
# Useful for spotting obvious mistakes and understanding model confidence.
test_batch = test_ds.take(1)
plt.figure(figsize=(10, 10))
for images, labels in test_batch:
    for i in range(min(6, len(images))):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        # Predict a single image: add a batch dimension with expand_dims.
        prediction = model.predict(tf.expand_dims(images[i].numpy(), 0))
        plt.title('Actual:' + class_names[labels[i].numpy()] +
                  '\nPredicted:{} {:.2f}%'.format(
                      class_names[np.argmax(prediction)],
                      100 * np.max(prediction)))
        plt.axis("off")
plt.show()
