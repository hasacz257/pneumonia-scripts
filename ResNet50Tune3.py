import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset_path = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray"
new_train_dir = f"{dataset_path}/new_train"
new_val_dir = f"{dataset_path}/new_val"
test_dir = f"{dataset_path}/test"


train_dataset = tf.keras.utils.image_dataset_from_directory(
    new_train_dir,
    image_size=(224, 224),
    batch_size=32
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    new_val_dir,
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


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3
)
base_model.trainable = False


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
history_extended_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    verbose=1
)


model.save("resnet50_extended_finetuned_model.keras")


test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Extended Fine-Tuned Test Loss: {test_loss}")
print(f"Extended Fine-Tuned Test Accuracy: {test_accuracy}")
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
plt.plot(history_extended_finetune.history['accuracy'], label='Extended Fine-Tune Train Accuracy')
plt.plot(history_extended_finetune.history['val_accuracy'], label='Extended Fine-Tune Val Accuracy', linestyle='dashed')


plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Extended Fine-Tuning Epochs')
plt.show()
plt.plot(history_extended_finetune.history['loss'], label='Extended Fine-Tune Train Loss')
plt.plot(history_extended_finetune.history['val_loss'], label='Extended Fine-Tune Val Loss', linestyle='dashed')


plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Extended Fine-Tuning Epochs')
plt.show()
