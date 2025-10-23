import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 1
BASE_LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 10  # Reduced epochs for quick testing

# Define the Dice coefficient metric
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Define the combined loss function (binary cross-entropy + Dice loss)
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1 - dice_coef(y_true, y_pred)
    return bce + dice_loss

# Build ResNet50 U-Net
def build_resnet50_unet():
    # Load ResNet50 as the encoder
    base_model = ResNet50(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the ResNet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Input layer
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Convert grayscale to RGB (ResNet50 expects 3 channels)
    grayscale_to_rgb = Conv2D(3, (1, 1), padding='same', name='grayscale_to_rgb')(inputs)

    # Encoder (ResNet50)
    encoder = base_model(grayscale_to_rgb)

    # Decoder
    x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(encoder)  # (14, 14, 512)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)  # (28, 28, 256)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)  # (56, 56, 128)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)  # (112, 112, 64)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Final upsampling to match input size
    x = UpSampling2D(size=(2, 2))(x)  # (224, 224, 64)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)  # (224, 224, 1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=BASE_LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coef, 'binary_accuracy', MeanIoU(num_classes=2)]
    )

    return model

# Data Generator for loading images and masks from the directory
def data_generator(image_paths, mask_paths, batch_size, augment=False):
    num_samples = len(image_paths)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_images = image_paths[offset:offset + batch_size]
            batch_masks = mask_paths[offset:offset + batch_size]

            # Load the images and masks
            batch_images = np.array([img_to_array(load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')) for img_path in batch_images])
            batch_masks = np.array([img_to_array(load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')) for mask_path in batch_masks])

            # Normalize the images and masks
            batch_images = batch_images / 255.0
            batch_masks = batch_masks / 255.0

            # Ensure the data is of type float32
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_masks = np.array(batch_masks, dtype=np.float32)

            yield batch_images, batch_masks

# Training and Evaluation Function
def train_and_evaluate_model(model, train_image_paths, train_mask_paths, val_image_paths, val_mask_paths, model_name, epochs=EPOCHS):
    model_path = os.path.join('models', f'{model_name}_best_model.h5')

    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_dice_coef', mode='max'),
        EarlyStopping(patience=5, monitor='val_dice_coef', mode='max', restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6, monitor='val_dice_coef', mode='max')
    ]

    train_gen = data_generator(train_image_paths, train_mask_paths, batch_size=BATCH_SIZE, augment=True)
    val_gen = data_generator(val_image_paths, val_mask_paths, batch_size=BATCH_SIZE, augment=False)

    train_steps = len(train_image_paths) // BATCH_SIZE
    val_steps = len(val_image_paths) // BATCH_SIZE

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    # Save the final model
    final_model_path = os.path.join('models', f'{model_name}_final_model.h5')
    model.save(final_model_path)

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['dice_coef'], label='Train Dice Coef')
    plt.plot(history.history['val_dice_coef'], label='Validation Dice Coef')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coef')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'{model_name}_training_history.png'))
    plt.show()

    return history

# Main Execution
if __name__ == "__main__":
    # Paths to your data (you need to replace these with your actual paths)
    train_images_path = 'COVID-QU-Ex/Infection_Segmentation_Data/Infection Segmentation Data/Train/COVID-19/images'
    train_masks_path = 'COVID-QU-Ex/Infection_Segmentation_Data/Infection Segmentation Data/Train/COVID-19/infection masks'
    val_images_path = 'COVID-QU-Ex/Infection_Segmentation_Data/Infection Segmentation Data/Val/COVID-19/images'
    val_masks_path = 'COVID-QU-Ex/Infection_Segmentation_Data/Infection Segmentation Data/Val/COVID-19/lung masks'

    # Get the image and mask file paths
    train_image_paths = [os.path.join(train_images_path, fname) for fname in os.listdir(train_images_path) if fname.endswith('.jpg') or fname.endswith('.png')]
    train_mask_paths = [os.path.join(train_masks_path, fname) for fname in os.listdir(train_masks_path) if fname.endswith('.jpg') or fname.endswith('.png')]

    val_image_paths = [os.path.join(val_images_path, fname) for fname in os.listdir(val_images_path) if fname.endswith('.jpg') or fname.endswith('.png')]
    val_mask_paths = [os.path.join(val_masks_path, fname) for fname in os.listdir(val_masks_path) if fname.endswith('.jpg') or fname.endswith('.png')]

    # Build and train the model
    model = build_resnet50_unet()
    model.summary()

    history = train_and_evaluate_model(
        model,
        train_image_paths,
        train_mask_paths,
        val_image_paths,
        val_mask_paths,
        model_name='resnet50_unet',
        epochs=EPOCHS
    )

    # Lung Segmentation Paths
    lung_train_images_path = 'COVID-QU-Ex/Lung_Segmentation_Data/Lung Segmentation Data/Train/images'
    lung_train_masks_path = 'COVID-QU-Ex/Lung_Segmentation_Data/Lung Segmentation Data/Train/masks'
    lung_val_images_path = 'COVID-QU-Ex/Lung_Segmentation_Data/Lung Segmentation Data/Val/images'
    lung_val_masks_path = 'COVID-QU-Ex/Lung_Segmentation_Data/Lung Segmentation Data/Val/masks'

    # Get the lung image and mask file paths
    lung_train_image_paths = [os.path.join(lung_train_images_path, fname) for fname in os.listdir(lung_train_images_path) if fname.endswith('.jpg') or fname.endswith('.png')]
    lung_train_mask_paths = [os.path.join(lung_train_masks_path, fname) for fname in os.listdir(lung_train_masks_path) if fname.endswith('.jpg') or fname.endswith('.png')]

    lung_val_image_paths = [os.path.join(lung_val_images_path, fname) for fname in os.listdir(lung_val_images_path) if fname.endswith('.jpg') or fname.endswith('.png')]
    lung_val_mask_paths = [os.path.join(lung_val_masks_path, fname) for fname in os.listdir(lung_val_masks_path) if fname.endswith('.jpg') or fname.endswith('.png')]

    # Build and train the lung segmentation model
    lung_model = build_resnet50_unet()
    lung_model.summary()

    lung_history = train_and_evaluate_model(
        lung_model,
        lung_train_image_paths,
        lung_train_mask_paths,
        lung_val_image_paths,
        lung_val_mask_paths,
        model_name='resnet50_unet_lung',
        epochs=EPOCHS
    )