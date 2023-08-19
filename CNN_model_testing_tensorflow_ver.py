import tensorflow as tf
import torch.cuda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

test_data_dir = "./data/test/"
train_data_dir = "./data/train/"

image_size = (224, 224)  # Resize images to this size

# Check if GPU is available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) == 0:
    raise RuntimeError("No GPU devices found.")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""

"""
# Define data generator with preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # Rescale pixel values between 0 and 1
    validation_split=0.2,
    rotation_range=20,       # Randomly rotate images within 20 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    horizontal_flip=True     # Randomly flip images horizontally
)

# Load and preprocess the training data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multiclass classification
    subset='training'
)

# Load and preprocess the validation data
validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multiclass classification
    subset='validation'
)

# Number of classes in your dataset
num_classes = len(train_generator.class_indices)

# Now you can use train_generator and validation_generator for training and validation
# Number of classes in your dataset
num_classes = len(train_generator.class_indices)

# Create the base ResNet50 model with pre-trained ImageNet weights
base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

# Add your custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create a TensorBoard callback
log_dir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[tensorboard_callback])

# Save the trained model
model.save('untrained_resnet_model.h5')