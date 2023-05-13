from mask_classifier import make_classifier
import tensorflow as tf
import os
import numpy as np
from sklearn.utils import shuffle
import cv2

tf.keras.backend.set_image_data_format('channels_first')
#######################################################

batch_size = 4

num_classes = 7
image_size = 224
WEIGHT_DECAY = 1e-4

num_of_epochs = 3
loss_file_path = 'mask_loss.txt'
val_loss_file_path = 'val_loss.txt'
model_save_path = 'final_weights.h5'
train_data_path = './data/EndoVis_2018/train/classifier_data'
val_data_path = './data/EndoVis_2018/val/classifier_data'
########################################################
train_images = []
train_mask = []
train_labels = []
val_images = []
val_mask = []
val_labels = []

for i in range(num_classes):
    current_class = os.path.join(train_data_path, str(i))
    samples = os.listdir(current_class)
    for s in samples:
        current_data_path = os.path.join(current_class, s)
        current_data = np.load(current_data_path)
        current_img = current_data[:, :, :3]
        current_mask = current_data[:, :, 3]
        current_image = current_img.copy().astype(np.float32)
        mean = [123.675, 116.28, 103.53]
        mean = np.asarray(mean)
        std = [58.395, 57.12, 57.375]
        std = np.asanyarray(std)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        cv2.subtract(current_image, mean, current_image)
        cv2.multiply(current_image, stdinv, current_image)
        current_image = np.transpose(current_image, [2, 0, 1])
        train_images.append(current_image)

        segm_mask = cv2.resize(current_mask, (image_size // 4, image_size // 4))
        segm_mask = segm_mask.astype(np.float32)
        segm_mask /= 255.0
        segm_mask = np.expand_dims(segm_mask, 0)
        train_mask.append(segm_mask)

        current_label = np.zeros((7,), dtype=np.float32)
        current_label[i] = 1
        train_labels.append(current_label)

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
train_mask = np.asarray(train_mask)

for i in range(num_classes):
    current_class = os.path.join(val_data_path, str(i))
    samples = os.listdir(current_class)
    for s in samples:
        current_data_path = os.path.join(current_class, s)
        current_data = np.load(current_data_path)
        current_img = current_data[:, :, :3]
        current_mask = current_data[:, :, 3]
        current_image = current_img.copy().astype(np.float32)
        mean = [123.675, 116.28, 103.53]
        mean = np.asarray(mean)
        std = [58.395, 57.12, 57.375]
        std = np.asanyarray(std)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        cv2.subtract(current_image, mean, current_image)
        cv2.multiply(current_image, stdinv, current_image)
        current_image = np.transpose(current_image, [2, 0, 1])
        val_images.append(current_image)

        segm_mask = cv2.resize(current_mask, (image_size // 4, image_size // 4))
        segm_mask = segm_mask.astype(np.float32)
        segm_mask /= 255.0
        segm_mask = np.expand_dims(segm_mask, 0)
        val_mask.append(segm_mask)

        current_label = np.zeros((7,), dtype=np.float32)
        current_label[i] = 1
        val_labels.append(current_label)

val_images = np.asarray(val_images)
val_labels = np.asarray(val_labels)
val_mask = np.asarray(val_mask)

print(train_images.shape)
print(train_mask.shape)
print(train_labels.shape)
print(val_images.shape)
print(val_mask.shape)
print(val_labels.shape)
# exit()

#######################################################
model = make_classifier(num_classes=num_classes, WEIGHT_DECAY=WEIGHT_DECAY)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)


@tf.function
def train_step(image, mask, label):
    with tf.GradientTape() as tape:
        op = model([image, mask, label], mode='margin', training=True)

        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_fn(label, op)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


@tf.function
def val_step(image, mask, label):
    op = model([image, mask], mode='softmax', training=False)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss = loss_fn(label, op)
    return loss


num_of_batches = train_images.shape[0] // batch_size
num_of_val_batches = val_images.shape[0] // batch_size
val_count = 1
best_val_loss = 1000.0
for epoch in range(num_of_epochs):
    this_epoch_loss = 0.0
    train_images, train_labels, train_mask = shuffle(train_images, train_labels, train_mask, random_state=666)
    for i in range(num_of_batches):
        image_batch = train_images[i * batch_size:i * batch_size + batch_size, :, :, :]
        label_batch = train_labels[i * batch_size:i * batch_size + batch_size]
        mask_batch = train_mask[i * batch_size:i * batch_size + batch_size]

        loss = train_step(image_batch, mask_batch, label_batch)
        print('Epoch ' + str(epoch) + '\tBatch ' + str(i) + '\tLoss: ' + str(loss) + '\n')
        this_epoch_loss += loss.numpy()

        if (i + 1) % 50 == 0:
            this_val_loss = 0.0
            for j in range(num_of_val_batches):
                image_batch = val_images[j * batch_size:j * batch_size + batch_size, :, :, :]
                label_batch = val_labels[j * batch_size:j * batch_size + batch_size]
                mask_batch = val_mask[j * batch_size:j * batch_size + batch_size]

                loss = val_step(image_batch, mask_batch, label_batch)

                loss = loss.numpy()
                this_val_loss += loss
            this_val_loss /= num_of_val_batches
            print('Validation: ' + str(val_count) + '\tLoss: ' + str(this_val_loss))
            val_count += 1
            with open(val_loss_file_path, 'a') as f:
                f.write(str(this_val_loss) + '\n')
            if this_val_loss <= best_val_loss:
                print('Model Loss improved from: ' + str(best_val_loss) + 'to: ' + str(
                    this_val_loss) + '\nSaving Weights\n')
                best_val_loss = this_val_loss
                model.save_weights(model_save_path)

    this_epoch_loss /= num_of_batches
    with open(loss_file_path, 'a') as f:
        f.write(str(this_epoch_loss) + '\n')
