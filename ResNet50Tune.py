import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
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




AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


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
    verbose=1
)
model.save("resnet50_initial_model.keras")


base_model.trainable = True 


for layer in base_model.layers[:-10]:
    layer.trainable = False


model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])


history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    verbose=1
)


model.save("resnet50_finetuned_model.keras")


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
plt.title('Accuracy over Epochs (Initial and Fine-Tuned)')
plt.show()


plt.plot(history_initial.history['loss'], label='Initial Train Loss')
plt.plot(history_initial.history['val_loss'], label='Initial Val Loss')


plt.plot(history_finetune.history['loss'], label='Fine-Tune Train Loss', linestyle='dashed')
plt.plot(history_finetune.history['val_loss'], label='Fine-Tune Val Loss', linestyle='dashed')


plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs (Initial and Fine-Tuned)')
plt.show()
