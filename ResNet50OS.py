import tensorflow as tf
import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
dataset_path = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray"
train_dir = f"{dataset_path}/new_train"
normal_dir = f"{train_dir}/NORMAL"
pneumonia_dir = f"{train_dir}/PNEUMONIA"
num_normal = len(os.listdir(normal_dir))
num_pneumonia = len(os.listdir(pneumonia_dir))
print(f"Number of NORMAL images: {num_normal}")
print(f"Number of PNEUMONIA images: {num_pneumonia}")
oversample_factor = num_pneumonia - num_normal
print(f"Number of additional NORMAL images needed: {oversample_factor}")
augmented_dir = f"{normal_dir}/augmented"
os.makedirs(augmented_dir, exist_ok=True)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
normal_images = [img for img in os.listdir(normal_dir) if os.path.isfile(os.path.join(normal_dir, img))]
random.shuffle(normal_images)


for i in range(oversample_factor):
    img_name = random.choice(normal_images)
    img_path = os.path.join(normal_dir, img_name)


    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)




    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=augmented_dir, save_prefix='aug', save_format='jpeg'):
        break


print("Oversampling completed!")
for file_name in os.listdir(augmented_dir):
    shutil.move(os.path.join(augmented_dir, file_name), normal_dir)
shutil.rmtree(augmented_dir)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32
)


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


train_dataset = train_dataset.map(normalize)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{dataset_path}/new_val",
    image_size=(224, 224),
    batch_size=32
)
val_dataset = val_dataset.map(normalize)


test_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{dataset_path}/test",
    image_size=(224, 224),
    batch_size=32
)
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


model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history_oversample = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    verbose=1
)


model.save("resnet50_oversampled_model.keras")


test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Oversampled Test Loss: {test_loss}")
print(f"Oversampled Test Accuracy: {test_accuracy}")
