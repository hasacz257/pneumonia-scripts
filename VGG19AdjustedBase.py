import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
dataset_path = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray"
train_dir = f"{dataset_path}/new_train"
val_dir = f"{dataset_path}/new_val"
test_dir = f"{dataset_path}/test"


batch_size = 32
img_size = (224, 224)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size, shuffle=True
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=batch_size, shuffle=False
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=img_size, batch_size=batch_size, shuffle=False
)


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


train_dataset = train_dataset.map(normalize)
val_dataset = val_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(AUTOTUNE)


base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -alpha * tf.pow((1 - pt), gamma) * tf.math.log(pt + tf.keras.backend.epsilon())
    return tf.reduce_mean(loss)


steps_per_epoch = len(train_dataset)
initial_lr = 1e-4
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=steps_per_epoch * 10,
    decay_rate=0.9,
    staircase=True
)


model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss=focal_loss,
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[early_stopping],
    verbose=1
)
test_loss, test_acc, test_auc = model.evaluate(test_dataset, verbose=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test AUC: {test_auc}")


test_preds = model.predict(test_dataset)
test_preds_rounded = np.round(test_preds)


y_true = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)


print("\nClassification Report:")
print(classification_report(y_true, test_preds_rounded, target_names=["NORMAL", "PNEUMONIA"]))


conf_matrix = confusion_matrix(y_true, test_preds_rounded)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()
plt.show()
