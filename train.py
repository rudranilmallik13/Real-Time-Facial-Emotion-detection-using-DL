# train.py
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from model import build_emotion_cnn
from utils import get_splits

def main(data_dir, epochs=30, batch_size=64):
    train_gen, val_gen, test_gen = get_splits(data_dir, batch_size=batch_size)

    model = build_emotion_cnn()
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint('emotion_model.h5', monitor='val_accuracy',
                                 verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early, reduce_lr]
    )

    # final evaluation
    loss, acc = model.evaluate(test_gen)
    print(f'Test loss: {loss:.4f}  Test acc: {acc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    main(args.data, epochs=args.epochs, batch_size=args.batch)