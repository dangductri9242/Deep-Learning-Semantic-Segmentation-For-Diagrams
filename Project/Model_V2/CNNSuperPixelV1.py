# Duc Tri Dang (Jin) - Jan 14th, 2025
# TADAI
# Updated version on Laura Paul's on April 26th, 2024.

#Tensorflow imports
import tensorflow as tf
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, CategoricalFocalCrossentropy
from keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
import sys

#basic unet model defined in separate file
from SuperPixelModelV1 import build_basic_spixelnet_normalized

#other imports (for visualization, etc.)
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import datetime
import seaborn as sns
#limit to CPU only - comment this out if you want GPU computation
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Set hyperparameters
EPOCHS = 128
BATCH_SIZE = 4
NUM_CLASSES = 5

#Given a dataset of images and the number of channels in the images (for RBG, it's 3; for labels, we're doing a one-hot encoding so it's 5)
#this function turns the images into a set of 128x128 patches
def get_patches(images: tf.Tensor, num_channels: int):
    #print(images.shape)
    print("extracting patches...")
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, 128, 128, 1],
        strides=[1, 128, 128, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    #print(patches.shape)
    print("reshaping patches") #patches must be reshaped into into the right dimensions for the dataset
    patch_dims = patches.shape[:3]
    patches = tf.reshape(patches, [np.prod(patch_dims), 128, 128, num_channels])
    print("patched!")
    #print(patches.shape)
    return patches

#used to return the rgb code of the colour we associated with each class label, given one pixel
def decode_pixel(encoding: list[float]) -> np.array:
    #print(encoding)
    encoding = np.argmax(encoding)
    if encoding == 1:
        return np.array([0,0,0]) #text
    elif encoding == 2:
        return np.array([47,79,79]) #edge
    elif encoding == 3:
        return np.array([143,188,143]) #node interior
    elif encoding == 4:
        return np.array([105,139,105]) #node outline
    else: 
        return np.array([255, 255, 255]) #background
    
#given an image, turn the class labels back to colours
def decode_label(image):
    img_array = np.zeros(shape=np.append(np.array(tf.shape(image)[:-1]), 3))
    img_shape = img_array.shape

    i = 0
    for encod_i in range(0, img_shape[0]):
        j = 0
        for encod_j in range(0, img_shape[1]):
            img_array[encod_i][encod_j] = decode_pixel(image[encod_i][encod_j])
            j+=1
        i+=1
        #print("i:", i, "j:", j)

    return img_array

#shows two images, intended to let you compare the results of prediction with the label
def show_image_and_mask(train_image, train_label,index):
    plt.figure(figsize=(5, 5))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    train_img_array = decode_label(train_image)
    train_lab_array = decode_label(train_label)

    #print(train_lab_array.astype(np.uint8))

    #print(train_label)
    #print(train_lab_array)
    train_im = PIL.Image.fromarray(train_img_array.astype(np.uint8)) #PIL fromarray() is limited in the data types it accepts
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_im)

    train_im.show()
    plt.show()
    plt.savefig("Super_Predicted"+str(index)+".png")

    label_im = PIL.Image.fromarray(train_lab_array.astype(np.uint8))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(label_im)
    label_im.show()
    plt.show()
    plt.savefig("Super_Label"+str(index)+".png")

print("loading the data...")
#Read in data
train_images_patches = pickle.load(open(os.path.expanduser('Data/data_10000/training_patches.pkl'), 'rb')) ##'rb' is 'read' and 'binary file'
train_labels_patches = pickle.load(open(os.path.expanduser('Data/data_10000/training_labels.pkl'), 'rb'))
val_images_patches = pickle.load(open(os.path.expanduser('Data/data_10000/validation_patches.pkl'), 'rb'))
val_labels_patches = pickle.load(open(os.path.expanduser('Data/data_10000/validation_labels.pkl'), 'rb'))

train_images_patches = train_images_patches / 255.0
val_images_patches = val_images_patches / 255.0
print("data loaded! starting preprocessing; shuffling ...")

#Put train images and labels in same order
permutation = np.random.permutation(train_images_patches.shape[0])
train_images_patches = train_images_patches[permutation]
train_labels_patches = train_labels_patches[permutation]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images_patches[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index

plt.show()

print("preprocessing; calculating class weights...")
print(train_labels_patches.shape)
print(np.count_nonzero(train_labels_patches[:, :, :, 0]))
print(np.count_nonzero(train_labels_patches[:, :, :, 1]))
print(np.count_nonzero(train_labels_patches[:, :, :, 2]))
print(np.count_nonzero(train_labels_patches[:, :, :, 3]))
print(np.count_nonzero(train_labels_patches[:, :, :, 4]))
class_weights = [0.5*np.prod(train_labels_patches[:, :, :, i].shape)/np.count_nonzero(train_labels_patches[:, :, :, i]) for i in range(NUM_CLASSES)]
print(class_weights)


#SuperPixel Model ==================================================================
model_super = build_basic_spixelnet_normalized()
print(model_super.summary())

#result = model_super.predict(train_images_patches[0:10])
#print(result)
#print(result.shape)

# Compile model
#must keep learning rate low, or the model errors out: https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
#may need to consider new loss function?
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=CategoricalCrossentropy(), metrics=['accuracy'])
#model_no_cut.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
#5e-7 is the standard
model_super.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=7e-8), loss=CategoricalFocalCrossentropy(alpha=class_weights, from_logits=False), metrics=['accuracy'])

print("starting model training...")

#print(model_no_cut(train_image_patches[:1]))
#print(train_label_patches[:1])

# Train model
#model.fit(train_image_patches, train_label_patches, epochs=EPOCHS, batch_size=BATCH_SIZE)
log_dir = "Logs/patchyCNN" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_V2_spx"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model_super.fit(train_images_patches, train_labels_patches, epochs=EPOCHS, validation_data=(val_images_patches, val_labels_patches), callbacks=[tensorboard_callback],batch_size = BATCH_SIZE)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="ModelCheckpoints/NO_CUT_checkpoint1.weights.h5", save_weights_only=True, verbose=1)

#print("sample output:\n", model_no_cut(tf.expand_dims(val_images[0], axis=0)))

for i in range(10):
    output = model_super(tf.expand_dims(val_images_patches[i], axis=0))
    #print(tf.reshape(output, tf.shape(output)[1:]))

    show_image_and_mask(tf.reshape(output, tf.shape(output)[1:]), val_labels_patches[i],i)
#plt.figure(figsize=(10,10))

#plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
#plt.legend(loc='lower right')
#plt.show()
# Evaluate accuracy on the validation set
print("eval:")
val_loss, val_acc = model_super.evaluate(val_images_patches, val_labels_patches)
print(val_loss, val_acc)

#predicting then pickle result`
data = "Result/SuperPixel_V2"
training_predict = model_super.predict(train_images_patches)
#pickle.dump(training_predict , open(os.path.join(data ,str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+"training_result.pkl"), "wb"))

val_predict = model_super.predict(val_images_patches)
#pickle.dump(training_predict , open(os.path.join(data ,str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+"validation_result.pkl"), "wb"))

def form_single_matrix(patch,predicted):
    a = [[0 for i in range(5)] for j in range(5)]
    shape = patch.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                truth_class = np.argmax(patch[i,j,k,:])
                predicted_class = np.argmax(predicted[i,j,k,:])
                a[truth_class][predicted_class] += 1
    return np.array(a)

def forming_matrix(train_labels_patches, val_labels_patches, predicted_train_labels_patchs, predicted_val_labels_patchs):
    confusion_matrix_train = form_single_matrix(train_labels_patches, predicted_train_labels_patchs)
    confusion_matrix_val = form_single_matrix(val_labels_patches, predicted_val_labels_patchs)

    return confusion_matrix_train, confusion_matrix_val

def accuracy_cal(matrix):
    true_value = matrix[0][0]+matrix[1][1]+matrix[2][2]+matrix[3][3]+matrix[4][4]
    all_value = sum(sum(matrix))
    return true_value/all_value

def save_pic(confusion_matrix_train, confusion_matrix_val, name):
    accuracy = accuracy_cal(confusion_matrix_train)
    print("Training:",accuracy)
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_matrix_train, annot=True, cmap='Blues',xticklabels=['Background', 'Text', 'Edges', 'Node Interior', 'Node Border'], yticklabels=['Background', 'Text', 'Edges', 'Node Interior', 'Node Border'])
    plt.title('Training Confusion Matrix '+name[:2]+" | Accuracy:"+str(int(accuracy*100))+"%")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig(name + '_train.png')

    accuracy = accuracy_cal(confusion_matrix_val)
    print("Validation:",accuracy)
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_matrix_val, annot=True, cmap='Blues',xticklabels=['Background', 'Text', 'Edges', 'Node Interior', 'Node Border'], yticklabels=['Background', 'Text', 'Edges', 'Node Interior', 'Node Border'])
    plt.title('Validation Confusion Matrix '+name[:2]+" | Accuracy:"+str(int(accuracy*100))+"%")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

train_matrix,val_matrix = forming_matrix(train_labels_patches, val_labels_patches, training_predict, val_predict)
save_pic(train_matrix,val_matrix,"V2_matrix"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

test_images_patches = pickle.load(open(os.path.expanduser('Data/data_10000/test_patches.pkl'), 'rb'))
test_labels_patches = pickle.load(open(os.path.expanduser('Data/data_10000/test_labels.pkl'), 'rb'))

test_images_patches = test_images_patches / 255.0

test_predict = model_super.predict(test_images_patches)
test_matrix = form_single_matrix(test_labels_patches,test_predict)

name = "V3_matrix"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
accuracy = accuracy_cal(test_matrix)
print("Testing:",accuracy)
plt.figure(figsize=(10,10))
sns.heatmap(test_matrix, annot=True, cmap='Blues',xticklabels=['Background', 'Text', 'Edges', 'Node Interior', 'Node Border'], yticklabels=['Background', 'Text', 'Edges', 'Node Interior', 'Node Border'])
plt.title('Testing Confusion Matrix '+name[:2]+" | Accuracy:"+str(int(accuracy*100))+"%")
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

plt.savefig(name + '_testing.png')
