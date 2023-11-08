import tensorflow as tf
import scipy
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import load_model


# Importing a Pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Creating a Custom Classification layer on top cause "include_top ='False'"
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
num_classes = 2
predictions = Dense(num_classes, activation='softmax')(x)

# Completing the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Define data preprocessing for testing
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
batch_size=32

# Create data generators for training and test data
train_generator = train_datagen.flow_from_directory(
    "D:/Project DD/dataset/train 1",
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical'
)
test_generator = test_datagen.flow_from_directory(
    "D:/Project DD/dataset/test 1",
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)
# loading model
model = load_model("model_version 2.h5")

# Training the model using the generator
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=1,
          validation_data=test_generator,
          validation_steps=test_generator.samples // batch_size)


# saving the model for later training
model.save_weights("model_weights_epoch V 2.1.h5")

# Saving the model with Arch
model.save("model_version 2.1.h5")
model.save('model_v2.1.keras')

# Evaluating the model (on test data)
test_loss, test_acc = model.evaluate(test_generator)
print("Test Accuracy",test_acc)