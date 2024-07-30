import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import pickle
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 16
IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
EPOCHS = 50
LEARNING_RATE = 0.003

def load_data_generator(dataset_dir: str, image_dim: Tuple[int, int], batch_size: int, is_training: bool = True):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        dataset_dir,
        shuffle=is_training,
        color_mode="rgb",
        class_mode="categorical",
        target_size=image_dim,
        batch_size=batch_size if is_training else 1
    )

def create_model(num_classes: int):
    feature_extractor = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4', 
                                       input_shape=INPUT_SHAPE, trainable=True)
    
    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.0005))
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# main exec
if __name__ == "__main__":
    train_dir = '/content/gdrive/MyDrive/augment_images/disease'
    validation_dir = '/content/gdrive/MyDrive/leaf_resized/disease'

    train_datagen = load_data_generator(train_dir, (IMG_HEIGHT, IMG_WIDTH), BATCH_SIZE)
    validation_datagen = load_data_generator(validation_dir, (IMG_HEIGHT, IMG_WIDTH), BATCH_SIZE, is_training=False)

    model = create_model(train_datagen.num_classes)
    model.summary()

    history = model.fit(
        train_datagen,
        epochs=EPOCHS,
        steps_per_epoch=train_datagen.samples // train_datagen.batch_size,
        validation_data=validation_datagen
    )

    model.save('/content/gdrive/MyDrive/saved_models_inc/model2')

    plot_results(history)

    # eval
    y_pred = model.predict(validation_datagen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print('Accuracy:', accuracy_score(validation_datagen.classes, y_pred_classes))
    plot_confusion_matrix(validation_datagen.classes, y_pred_classes, validation_datagen.class_indices.keys())
