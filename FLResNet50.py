import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    """
    Focal Loss for binary classification.
    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities (from sigmoid).
        gamma: Focusing parameter.
        alpha: Balancing parameter.
    Returns:
        Loss value.
    """
    y_true = tf.cast(y_true, tf.float32)
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_weight = tf.pow(1 - y_pred, gamma) * y_true + tf.pow(y_pred, gamma) * (1 - y_true)
    loss = -alpha_factor * focal_weight * tf.math.log(y_pred + tf.keras.backend.epsilon())
    return tf.reduce_mean(loss)


dataset_path = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray"
train_dir = f"{dataset_path}/new_train"
val_dir = f"{dataset_path}/new_val"
test_dir = f"{dataset_path}/test"


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32
)


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


train_dataset = train_dataset.map(normalize)
val_dataset = val_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)


base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False




model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=lambda y_true, y_pred: focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)


history_focal = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    verbose=1
)
model.save("resnet50_custom_focal_loss_model.keras")
test_loss_focal, test_accuracy_focal = model.evaluate(test_dataset)
print(f"Focal Loss Test Loss: {test_loss_focal}")
print(f"Focal Loss Test Accuracy: {test_accuracy_focal}")
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = np.round(model.predict(test_dataset))


print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
plt.plot(history_focal.history['accuracy'], label='Train Accuracy')
plt.plot(history_focal.history['val_accuracy'], label='Validation Accuracy', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs with Custom Focal Loss')
plt.show()
plt.plot(history_focal.history['loss'], label='Train Loss')
plt.plot(history_focal.history['val_loss'], label='Validation Loss', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs with Custom Focal Loss')
plt.show()
