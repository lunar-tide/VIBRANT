import argparse
import os
import pickle
import datetime
from BinSim3D import SiameseVideoModel
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tools.drawpic import plot_and_save_metrics


def load_data(filename):
    with open(filename, 'rb') as file:
        video_data = pickle.load(file)
    return video_data


def parse_tfrecord_fn(example):
    feature_description = {
        'array1': tf.io.FixedLenFeature([], tf.string),
        'array2': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)

    array1 = tf.io.parse_tensor(example['array1'], out_type=tf.float32)
    array2 = tf.io.parse_tensor(example['array2'], out_type=tf.float32)
    label = example['label'][0]

    return (array1, array2), label

def train_large_data(train_data_path, val_data_path, batch_size=8):
    train_tfrecord_files = [os.path.join(train_data_path, filename) for filename in os.listdir(train_data_path) if
                      filename.endswith('.tfrecord')]
    val_tfrecord_files = [os.path.join(val_data_path, filename) for filename in os.listdir(val_data_path) if
                      filename.endswith('.tfrecord')]
    train_dataset = tf.data.TFRecordDataset(train_tfrecord_files)
    train_dataset = train_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.TFRecordDataset(val_tfrecord_files)
    val_dataset = val_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


    model = SiameseVideoModel(frame_count=50, height=256, width=256, channels=1)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"siamese_video_model_{current_time}.h5"

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # 保存最佳模型
    model_checkpoint = ModelCheckpoint(
        model_filename,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.model.fit(train_dataset, validation_data=val_dataset, epochs=200,callbacks=[early_stopping, model_checkpoint,reduce_lr])
    plot_and_save_metrics(history)

def main():
    args = parse_args()
    train_large_data(args.train_data, args.val_data)

def parse_args():
    parser = argparse.ArgumentParser(description="VideBinDiff")
    parser.add_argument('--train-data', type=str, default=" ", help="Train data storage path.")
    parser.add_argument('--val-data', type=str, default=" ", help="Validation data storage path.")
    return parser.parse_args()

if __name__ == "__main__":
    main()