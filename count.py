import os


train_normal_dir = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray\train\NORMAL"
train_pneumonia_dir = r"C:\Users\zzblo\OneDrive\Desktop\Uni\Pneumonia\data\chest_xray\train\PNEUMONIA"


normal_count = len(os.listdir(train_normal_dir))
pneumonia_count = len(os.listdir(train_pneumonia_dir))


print(f"Normal images: {normal_count}")
print(f"Pneumonia images: {pneumonia_count}")
print(f"Total training images: {normal_count + pneumonia_count}")
