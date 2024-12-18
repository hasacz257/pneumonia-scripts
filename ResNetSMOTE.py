import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset_path = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray"
train_dir = f"{dataset_path}/new_train"
val_dir = f"{dataset_path}/new_val"
test_dir = f"{dataset_path}/test"


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(224, 224), batch_size=32
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=(224, 224), batch_size=32
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=(224, 224), batch_size=32
)


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


train_dataset = train_dataset.map(normalize)
val_dataset = val_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)


# Step 2: Extract Features Using Pre-trained ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)


def extract_features(dataset):
    features = []
    labels = []
    for images, lbls in dataset:
        feats = feature_extractor.predict(images, verbose=0)
        features.append(feats)
        labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


train_features, train_labels = extract_features(train_dataset)
val_features, val_labels = extract_features(val_dataset)
test_features, test_labels = extract_features(test_dataset)
smote = SMOTE(random_state=42)
train_features_smote, train_labels_smote = smote.fit_resample(train_features, train_labels)
classifier = tf.keras.Sequential([
    Dense(128, activation="relu", input_shape=(train_features_smote.shape[1],)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])


classifier.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])


history = classifier.fit(
    train_features_smote, train_labels_smote,
    validation_data=(val_features, val_labels),
    epochs=20,
    batch_size=32,
    verbose=1
)
test_loss, test_accuracy = classifier.evaluate(test_features, test_labels)
print(f"SMOTE Test Loss: {test_loss}")
print(f"SMOTE Test Accuracy: {test_accuracy}")


test_predictions = np.round(classifier.predict(test_features))


print("\nClassification Report:")
print(classification_report(test_labels, test_predictions, target_names=["NORMAL", "PNEUMONIA"]))


conf_matrix = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
