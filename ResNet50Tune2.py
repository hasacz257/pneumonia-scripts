import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt


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


data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])


train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


labels = np.concatenate([y for x, y in train_dataset], axis=0)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class Weights: {class_weights_dict}")


base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False 
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


history_initial = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights_dict,
    verbose=1
)


model.save("resnet50_augmented_initial_model.keras")


base_model.trainable = True


for layer in base_model.layers[:-10]:
    layer.trainable = False


model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])


history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights_dict,
    verbose=1
)


model.save("resnet50_augmented_finetuned_model.keras")


test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Fine-Tuned Test Loss: {test_loss}")
print(f"Fine-Tuned Test Accuracy: {test_accuracy}")


plt.plot(history_initial.history['accuracy'], label='Initial Train Accuracy')
plt.plot(history_initial.history['val_accuracy'], label='Initial Val Accuracy')


plt.plot(history_finetune.history['accuracy'], label='Fine-Tune Train Accuracy', linestyle='dashed')
plt.plot(history_finetune.history['val_accuracy'], label='Fine-Tune Val Accuracy', linestyle='dashed')


plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs (Augmented & Fine-Tuned)')
plt.show()




plt.plot(history_initial.history['loss'], label='Initial Train Loss')
plt.plot(history_initial.history['val_loss'], label='Initial Val Loss')


plt.plot(history_finetune.history['loss'], label='Fine-Tune Train Loss', linestyle='dashed')
plt.plot(history_finetune.history['val_loss'], label='Fine-Tune Val Loss', linestyle='dashed')


plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs (Augmented & Fine-Tuned)')
plt.show()
