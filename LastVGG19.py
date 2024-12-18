import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


base_path = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray"
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "new_val")
test_dir = os.path.join(base_path, "test")


batch_size = 32
img_size = (224, 224)
initial_lr = 3e-5
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


train_dataset = train_dataset.map(normalize)
val_dataset = val_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05)
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(AUTOTUNE)




train_labels = []
for _, lbls in train_dataset.take(-1):
    train_labels.extend(lbls.numpy())
train_labels = np.array(train_labels)


class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Class Weights:", class_weights)


base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


for layer in base_model.layers[:-5]:
    layer.trainable = False
for layer in base_model.layers[-5:]:
    layer.trainable = True


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.35):
    y_true = tf.cast(y_true, tf.float32)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -alpha * tf.pow((1 - pt), gamma) * tf.math.log(pt + tf.keras.backend.epsilon())
    return tf.reduce_mean(loss)


steps_per_epoch = len(train_dataset)
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
    class_weight=class_weights,
    verbose=1
)


test_loss, test_acc, test_auc = model.evaluate(test_dataset, verbose=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test AUC: {test_auc}")


test_preds = model.predict(test_dataset)
y_true = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
possible_thresholds = np.linspace(0.1, 0.9, 9)
best_recall = 0.0
best_thresh = 0.5


for thresh in possible_thresholds:
    test_preds_rounded = (test_preds > thresh).astype(int)
    from sklearn.metrics import classification_report
    report = classification_report(y_true, test_preds_rounded, target_names=["NORMAL", "PNEUMONIA"], output_dict=True)
    normal_recall = report["NORMAL"]["recall"]
    if normal_recall > best_recall:
        best_recall = normal_recall
        best_thresh = thresh


print(f"Best threshold for NORMAL recall: {best_thresh}")


test_preds_rounded = (test_preds > best_thresh).astype(int)
print("Classification Report (with adjusted threshold):")
print(classification_report(y_true, test_preds_rounded, target_names=["NORMAL", "PNEUMONIA"]))


conf_matrix = confusion_matrix(y_true, test_preds_rounded)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix with Adjusted Threshold")
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
