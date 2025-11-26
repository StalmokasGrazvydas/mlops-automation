import os, argparse, json
import tensorflow as tf
from utils import list_images, label_from_name

def make_dataset(paths, labels, batch=32, img_size=(64,64)):
    paths = tf.constant(paths)
    labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, y
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)\
             .shuffle(buffer_size=min(1000, len(paths)))\
             .batch(batch).prefetch(tf.data.AUTOTUNE)

def build_model(num_classes):
    m = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(64,64,3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_folder", required=True)
    ap.add_argument("--testing_folder",  required=True)
    ap.add_argument("--output_folder",   required=True)
    ap.add_argument("--epochs",          type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    train_files = list_images(args.training_folder)
    test_files  = list_images(args.testing_folder)
    if not train_files or not test_files:
        raise RuntimeError("No images found in training/testing folders.")

    classes = sorted({label_from_name(p) for p in train_files})
    class_to_idx = {c:i for i,c in enumerate(classes)}
    y_train = [class_to_idx[label_from_name(p)] for p in train_files]
    y_test  = [class_to_idx.get(label_from_name(p), -1) for p in test_files]
    if any(y < 0 for y in y_test):
        raise RuntimeError("Testing set contains unknown classes.")

    ds_train = make_dataset(train_files, y_train)
    ds_test  = make_dataset(test_files,  y_test)

    model = build_model(num_classes=len(classes))
    hist = model.fit(ds_train, validation_data=ds_test, epochs=args.epochs)

    loss, acc = model.evaluate(ds_test, verbose=0)
    metrics = {"val_accuracy": float(acc), "val_loss": float(loss), "classes": classes}
    with open(os.path.join(args.output_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    model.save(os.path.join(args.output_folder, "model.keras"))
    print("TRAINED_CLASSES=", classes)
    print("VAL_ACCURACY=", acc)

if __name__ == "__main__":
    main()
