import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import shutil




dataset_path = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray"
train_dir = f"{dataset_path}/train"
test_dir = f"{dataset_path}/test"




new_train_dir = f"{dataset_path}/new_train"
new_val_dir = f"{dataset_path}/new_val"


def split_data(source_dir, train_dir, val_dir, split_ratio=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)


    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)


        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)


        
        file_names = os.listdir(class_path)
        train_files, val_files = train_test_split(file_names, test_size=split_ratio, random_state=42)


      
        for file in train_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(train_class_dir, file))
        for file in val_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(val_class_dir, file))




split_data(train_dir, new_train_dir, new_val_dir, split_ratio=0.2)




train_dataset = tf.keras.utils.image_dataset_from_directory(
    new_train_dir,
    image_size=(128, 128),
    batch_size=32
)


val_dataset = tf.keras.utils.image_dataset_from_directory(
    new_val_dir,
    image_size=(128, 128),
    batch_size=32
)




test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
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




print(f"Train batches: {len(train_dataset)}")
print(f"Validation batches: {len(val_dataset)}")
print(f"Test batches: {len(test_dataset)}")


for image_batch, label_batch in train_dataset.take(1):
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Label batch shape: {label_batch.shape}")
