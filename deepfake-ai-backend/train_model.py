import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print(f"TensorFlow Version: {tf.__version__}")

def build_spatiotemporal_model():
    """
    Builds the ResNet-50 + LSTM Hybrid Model for Deepfake Detection.
    Input: Tensors of shape (Batch Size, 30 frames, 224 px, 224 px, 3 channels)
    """
    # 1. Define the input shape (30 frames of 224x224 RGB images)
    video_input = Input(shape=(30, 224, 224, 3), name='video_sequence_input')

    # 2. Load ResNet-50 (Spatial Feature Extractor)
    # include_top=False strips the ImageNet classifier, keeping the base convolutional layers
    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Freeze the pre-trained ResNet layers initially so we don't destroy weights during early training
    resnet_base.trainable = False

    # 3. Apply ResNet-50 to each of the 30 frames independently
    encoded_frames = TimeDistributed(resnet_base, name='spatial_feature_extractor')(video_input)

    # 4. Temporal Sequence Analyzer (LSTM)
    # Processes the sequence of 30 spatial feature vectors to spot temporal inconsistencies/jitter
    lstm_out = LSTM(128, return_sequences=False, name='temporal_analyzer')(encoded_frames)

    # 5. Classification Head
    x = Dense(512, activation='relu', name='dense_classifier')(lstm_out)
    x = Dropout(0.5, name='dropout_regularization')(x) # Helps prevent overfitting
    
    # Final output layer: Sigmoid activation for Binary Classification (0.0 Real -> 1.0 Fake)
    output = Dense(1, activation='sigmoid', name='final_output')(x)

    # Compile the model
    model = Model(inputs=video_input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4), # Lower learning rate for stable convergence
        loss='binary_crossentropy',         # Standard loss for binary real vs fake classification
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Instantiate and check model layout
if __name__ == "__main__":
    deepfake_validator = build_spatiotemporal_model()
    deepfake_validator.summary()
    
    # --- Define Callbacks for Training ---
    callbacks = [
        # Automatically saves the best weights based on validation accuracy
        ModelCheckpoint(
            filepath='best_deepfake_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Halts training early if the validation loss fails to improve for 10 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n[INFO] Spatiotemporal model compiled and ready for training datasets.")