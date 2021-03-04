########### Basic Parameters for Running: ################################
    
Version = "7.2.1-Small-Diego"                      # Used for tflite Filename
Testing_Percentage = 0.2                # 0.0 = Use all Images for Training
Training_Percentage = 0.2              # 0.0 = Use all Images for Training
Epoch_Anz = 200

##########################################################################


import tensorflow as tf
import autokeras as ak
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
from contextlib import redirect_stdout
from sklearn.utils import shuffle
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, auc, cohen_kappa_score, classification_report
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.utils import to_categorical
from PIL import Image

loss_ges = np.array([])
val_loss_ges = np.array([])

#%matplotlib inline
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

##########################################################################

def generator_to_array(generator,
                       X):
    """
    given a list of tuples with data aumentation, tranform it to and array of X's values or y's values

    Arguments
    ---------
    generator:    generator from tf.keras.preprocessing.image.ImageDataGenerator.flow

    X:            boolean. If True, extract the X's values.
                           If False, extract the y's values.
    """
    if X:
        array = generator[0][0]
        for i in range(1, len(generator)):
            array = np.append(array, generator[i][0], axis=0)
    else:
        array = generator[0][1]
        for i in range(1, len(generator)):
            array = np.append(array, generator[i][1], axis=0)
    return array

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          file_name='Confusion matrix'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(file_name + '.png')
    #plt.show()

##########################################################################

Input_dir='dataset_scut/easy'

Batch_Size = 4
Shift_Range = 1
Brightness_Range = 0.3
Rotation_Angle = 10
ZoomRange = 0.4

files = glob.glob(Input_dir + '/*.*')
x_data = []
y_data = []

subdir = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]

for aktsubdir in subdir:
    files = glob.glob(Input_dir + '/' + aktsubdir + '/*.jpg')
    if aktsubdir == "NaN":
        category = 10                # NaN does not work --> convert to 10
    else:
        category = aktsubdir
    
    for aktfile in files:
        test_image = Image.open(aktfile)
        test_image = np.array(test_image, dtype="float32")
        x_data.append(test_image)
        y_data.append(np.array([category]))

x_data = np.array(x_data)
y_data = np.array(y_data)
y_data = to_categorical(y_data, 20)
print(x_data.shape)
print(y_data.shape)

x_data, y_data = shuffle(x_data, y_data)

datagen = ImageDataGenerator(width_shift_range=[-Shift_Range,Shift_Range], 
                             height_shift_range=[-Shift_Range,Shift_Range],
                             brightness_range=[1-Brightness_Range,1+Brightness_Range],
                             zoom_range=[1-ZoomRange, 1+ZoomRange],
                             rotation_range=Rotation_Angle,
                             validation_split=Testing_Percentage)
train_temp_iterator = datagen.flow(x_data, y_data, batch_size=Batch_Size, subset='training')
test_iterator = datagen.flow(x_data, y_data, batch_size=Batch_Size, subset='validation')
X_train_temp = generator_to_array(train_temp_iterator, True)
y_train_temp = generator_to_array(train_temp_iterator, False)
X_test = generator_to_array(test_iterator, True)
y_test = generator_to_array(test_iterator, False)
if (Training_Percentage > 0):
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=Training_Percentage)

##########################################################################

model = ak.ImageClassifier(overwrite=True, multi_label=True, max_trials=100)

##########################################################################

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit(x=X_train, y=y_train, validation_data = (X_val, y_val), epochs = Epoch_Anz, callbacks = [early_stop])

##########################################################################

if (Testing_Percentage > 0):
    y_pred = model.predict(X_test)
    print('Accuracy: ' + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))))
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    plot_confusion_matrix(cm=confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)), 
                      normalize    = False,
                      target_names = subdir,
                      title        = "Confusion Matrix",
                      file_name    = "testCF")

##########################################################################

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.export_model().summary()

##########################################################################

model.export_model().save('watermetter_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
flatbuffer_size = open('watermetter_model.tflite', "wb").write(tflite_model)

##########################################################################

interpreter = tf.lite.Interpreter(model_path='watermetter_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
for i in range(0, len(X_test)):
    interpreter.set_tensor(input_details[0]['index'], [X_test[i]])
    interpreter.invoke()
    if i == 0:
        y_pred_lite = interpreter.get_tensor(output_details[0]['index'])
    else:
        y_pred_lite = np.append(y_pred_lite, interpreter.get_tensor(output_details[0]['index']), axis=0)

print('Accuracy: ' + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_lite, axis=1))))
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred_lite, axis=1)))
plot_confusion_matrix(cm=confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred_lite, axis=1)), 
                      normalize    = False,
                      target_names = subdir,
                      title        = "Confusion Matrix",
                      file_name    = 'tfliteTestCF')