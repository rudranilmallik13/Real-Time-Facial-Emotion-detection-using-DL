EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}
# ...existing code...

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_splits(data_dir, img_size=(48, 48), batch_size=64):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        f"{data_dir}/val",
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    test_gen = datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    return train_gen, val_gen, test_gen