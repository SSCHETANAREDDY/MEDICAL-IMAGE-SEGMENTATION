import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, concatenate, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape, Dense, Multiply, Permute, multiply
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU, BinaryAccuracy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import cv2
from pathlib import Path



# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3  # RGB
BATCH_SIZE = 16
EPOCHS = 2
BASE_LEARNING_RATE = 0.001
SEED = 42

# Create output directories if they don't exist
output_dirs = ['results', 'models', 'figures']
for dir_name in output_dirs:
    os.makedirs(dir_name, exist_ok=True)

# Define metrics and loss functions
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

# Data loading functions
def load_data(base_path, segment_type="lung"):
    data = {
        'train': {'images': [], 'masks': []},
        'val': {'images': [], 'masks': []},
        'test': {'images': [], 'masks': []}
    }

    if segment_type == "lung":
        data_path = os.path.join(base_path, "Lung_Segmentation_Data", "Lung Segmentation Data")
    else:  # infection
        data_path = os.path.join(base_path, "Infection_Segmentation_Data", "Infection segmentation Data")

    classes = ["COVID_19", "Non-COVID", "Normal"]

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split.capitalize())

        for class_name in classes:
            class_path = os.path.join(split_path, class_name)

            if segment_type == "lung":
                img_path = os.path.join(class_path, "Images")
                mask_path = os.path.join(class_path, "lung masks")
            else:  # infection
                img_path = os.path.join(class_path, "images")
                mask_path = os.path.join(class_path, "infection masks")

            if os.path.exists(img_path) and os.path.exists(mask_path):
                image_files = sorted(glob.glob(os.path.join(img_path, "*.png")))

                for img_file in image_files:
                    filename = os.path.basename(img_file)
                    mask_file = os.path.join(mask_path, filename)

                    if os.path.exists(mask_file):
                        data[split]['images'].append(img_file)
                        data[split]['masks'].append(mask_file)

    print(f"Loaded {segment_type} segmentation data:")
    print(f"Train: {len(data['train']['images'])} images")
    print(f"Validation: {len(data['val']['images'])} images")
    print(f"Test: {len(data['test']['images'])} images")

    return data

# Data preprocessing functions
def preprocess_image(image_path, mask_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (224, 224))
    mask = cv2.resize(mask, (224, 224))

    img = np.stack((img,) * 3, axis=-1)
    img = (img / 255.0).astype(np.float32)
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return img, mask

def data_generator(image_paths, mask_paths, batch_size=BATCH_SIZE, augment=False):
    num_samples = len(image_paths)
    indices = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_images = []
            batch_masks = []

            for idx in batch_indices:
                img, mask = preprocess_image(image_paths[idx], mask_paths[idx])

                if img is None or mask is None:
                    continue

                if augment:
                    if np.random.random() > 0.5:
                        img = np.fliplr(img)
                        mask = np.fliplr(mask)

                    angle = np.random.randint(-10, 10)
                    if angle != 0:
                        M = cv2.getRotationMatrix2D((224 // 2, 224 // 2), angle, 1)
                        img = cv2.warpAffine(img, M, (224, 224))
                        mask = cv2.warpAffine(mask, M, (224, 224))
                        mask = (mask > 0.5).astype(np.float32)
                        mask = np.expand_dims(mask, axis=-1)

                batch_images.append(img)
                batch_masks.append(mask)

            batch_images = np.array(batch_images)
            batch_masks = np.array(batch_masks)

            yield batch_images, batch_masks

# Model 2: ResUNet (U-Net with Residual Blocks)
def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    res = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    res = BatchNormalization()(res)
    res = Activation('relu')(res)

    res = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(res)
    res = BatchNormalization()(res)

    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    output = Add()([res, shortcut])
    output = Activation('relu')(output)

    return output

def attention_gate(x, g, filters):
    theta_x = Conv2D(filters, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)

    # Upsample theta_x to match phi_g's spatial dimensions
    theta_x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(theta_x)

    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, (1, 1), padding='same')(f)
    att_map = Activation('sigmoid')(psi_f)

    y = multiply([x, att_map])

    return y

def build_resunet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Encoder
    c1 = Conv2D(32, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = residual_block(c1, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = residual_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = residual_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bridge
    b = residual_block(p4, 512)

    # Decoder with attention gates
    u4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(b)
    a4 = attention_gate(c4, u4, 256)
    u4 = concatenate([u4, a4])
    u4 = residual_block(u4, 256)

    u3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(u4)
    a3 = attention_gate(c3, u3, 128)
    u3 = concatenate([u3, a3])
    u3 = residual_block(u3, 128)

    u2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u3)
    a2 = attention_gate(c2, u2, 64)
    u2 = concatenate([u2, a2])
    u2 = residual_block(u2, 64)

    u1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(u2)
    a1 = attention_gate(c1, u1, 32)
    u1 = concatenate([u1, a1])
    u1 = residual_block(u1, 32)

    # Final output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u1)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile model with custom loss
    model.compile(
        optimizer=Adam(learning_rate=BASE_LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coef, 'binary_accuracy', MeanIoU(num_classes=2)]
    )

    return model

# Model 3: ResNet50 for Segmentation
def build_resnet50_unet():
    base_model = ResNet50(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    grayscale_to_rgb = Conv2D(3, (1, 1), padding='same')(inputs)
    encoder = base_model(grayscale_to_rgb)

    skip1 = base_model.get_layer('conv1_relu').output
    skip2 = base_model.get_layer('conv2_block3_out').output
    skip3 = base_model.get_layer('conv3_block4_out').output
    skip4 = base_model.get_layer('conv4_block6_out').output

    x = encoder

    x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x, skip4])
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x, skip3])
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x, skip2])
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x, skip1])
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=BASE_LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coef, 'binary_accuracy', MeanIoU(num_classes=2)]
    )

    return model

# Training and evaluation function
def train_and_evaluate_model(model, data, model_name, segment_type, epochs=EPOCHS):
    model_path = os.path.join('models', f'{model_name}_{segment_type}_best_model.h5')

    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_dice_coef', mode='max'),
        EarlyStopping(patience=15, monitor='val_dice_coef', mode='max', restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, monitor='val_dice_coef', mode='max')
    ]

    train_gen = data_generator(
        data['train']['images'],
        data['train']['masks'],
        batch_size=BATCH_SIZE,
        augment=True
    )

    val_gen = data_generator(
        data['val']['images'],
        data['val']['masks'],
        batch_size=BATCH_SIZE,
        augment=False
    )

    train_steps = len(data['train']['images']) // BATCH_SIZE
    val_steps = len(data['val']['images']) // BATCH_SIZE

    train_steps = max(1, train_steps)
    val_steps = max(1, val_steps)

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    final_model_path = os.path.join('models', f'{model_name}_{segment_type}_final_model.h5')
    model.save(final_model_path)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title(f'{model_name} - Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title(f'{model_name} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    history_path = os.path.join('figures', f'{model_name}_{segment_type}_training_history.png')
    plt.savefig(history_path)
    plt.close()

    test_gen = data_generator(
        data['test']['images'],
        data['test']['masks'],
        batch_size=BATCH_SIZE,
        augment=False
    )

    test_steps = max(1, len(data['test']['images']) // BATCH_SIZE)

    model.load_weights(model_path)

    test_results = model.evaluate(test_gen, steps=test_steps)

    metrics = {
        'dice_coef': test_results[1],
        'binary_accuracy': test_results[2],
        'mean_iou': test_results[3],
        'history': history.history
    }

    return metrics

def run_experiment(covid_qu_ex_path=None, reduced_epochs=3):  # Set reduced_epochs to 3
    if covid_qu_ex_path is None:
        covid_qu_ex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'COVID-QU-Ex')

    results_df = pd.DataFrame(columns=[  # Set up an empty dataframe to store results
        'Model', 'Segmentation Type', 'Dice Coefficient', 'Binary Accuracy', 'Mean IoU'
    ])

    model_configs = [  # Specify the models you want to run
        ('ResUNet', build_resunet),
        ('ResNet50', build_resnet50_unet)
    ]

    print("\n=== Loading Lung Segmentation Data ===")
    lung_data = load_data(covid_qu_ex_path, segment_type="lung")

    for model_name, model_builder in model_configs:
        print(f"\n=== Training {model_name} for Lung Segmentation ===")
        model = model_builder()
        metrics = train_and_evaluate_model(
            model,
            lung_data,
            model_name,
            'lung',
            epochs=reduced_epochs  # This ensures only 3 epochs for the lung data
        )

        results_df = pd.concat([results_df, pd.DataFrame({
            'Model': [model_name],
            'Segmentation Type': ['Lung'],
            'Dice Coefficient': [metrics['dice_coef']],
            'Binary Accuracy': [metrics['binary_accuracy']],
            'Mean IoU': [metrics['mean_iou']]
        })], ignore_index=True)

    print("\n=== Loading Infection Segmentation Data ===")
    infection_data = load_data(covid_qu_ex_path, segment_type="infection")

    for model_name, model_builder in model_configs:
        print(f"\n=== Training {model_name} for Infection Segmentation ===")
        model = model_builder()
        metrics = train_and_evaluate_model(
            model,
            infection_data,
            model_name,
            'infection',
            epochs=reduced_epochs  # This ensures only 3 epochs for the infection data
        )

        results_df = pd.concat([results_df, pd.DataFrame({
            'Model': [model_name],
            'Segmentation Type': ['Infection'],
            'Dice Coefficient': [metrics['dice_coef']],
            'Binary Accuracy': [metrics['binary_accuracy']],
            'Mean IoU': [metrics['mean_iou']]
        })], ignore_index=True)

    results_path = os.path.join('results', 'segmentation_results.csv')
    results_df.to_csv(results_path, index=False)  # Save the results to CSV

    print("\n=== Final Results ===")
    print(results_df)

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    for seg_type in ['Lung', 'Infection']:
        seg_results = results_df[results_df['Segmentation Type'] == seg_type]
        plt.bar(
            [f"{model} ({seg_type})" for model in seg_results['Model']],
            seg_results['Dice Coefficient'],
            alpha=0.7,
            label=seg_type
        )
    plt.title('Dice Coefficient Comparison')
    plt.ylabel('Dice Coefficient')
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(3, 1, 2)
    for seg_type in ['Lung', 'Infection']:
        seg_results = results_df[results_df['Segmentation Type'] == seg_type]
        plt.bar(
            [f"{model} ({seg_type})" for model in seg_results['Model']],
            seg_results['Binary Accuracy'],
            alpha=0.7,
            label=seg_type
        )
    plt.title('Binary Accuracy Comparison')
    plt.ylabel('Binary Accuracy')
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(3, 1, 3)
    for seg_type in ['Lung', 'Infection']:
        seg_results = results_df[results_df['Segmentation Type'] == seg_type]
        plt.bar(
            [f"{model} ({seg_type})" for model in seg_results['Model']],
            seg_results['Mean IoU'],
            alpha=0.7,
            label=seg_type
        )
    plt.title('Mean IoU Comparison')
    plt.ylabel('Mean IoU')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'model_comparison.png'))
    plt.close()

    print(f"Results saved to {results_path}")
    print(f"Comparison chart saved to {os.path.join('figures', 'model_comparison.png')}")

# Main execution
if __name__ == "__main__":
    run_experiment()
