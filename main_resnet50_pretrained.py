# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


NUM_CLASSES = 4
IMAGE_RESIZE = 224
NUM_EPOCHS = 1000
EARLY_STOP_PATIENCE = 20 # EARLY_STOP_PATIENCE must be < NUM_EPOCHS
BATCH_SIZE = 32
lr = 0.001  # learning rate

InceptionResNetV2_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy'] # Common accuracy metric for all outputs, but can use different metrics for different output


trImgNum = 6776
valImgNum = 1694


fname, accuracy, AUCs, = [], [], []
recall_macro, precision_macro, f1score_macro = [], [], []
recall_micro, precision_micro, f1score_micro = [], [], []
recall_weighted, precision_weighted, f1score_weighted = [], [], []

for i in range(1,13):
        
    leadFolder = 'lead'+str(i)

    
    # These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
    # Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
    STEPS_PER_EPOCH_TRAINING = trImgNum//BATCH_SIZE
    STEPS_PER_EPOCH_VALIDATION = valImgNum//BATCH_SIZE
    
    # Using 1 to easily manage mapping between test_generator & prediction for submission preparation
    BATCH_SIZE_TESTING = 1
    
    
    ## data load
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
#    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    image_size = IMAGE_RESIZE
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    tr_dir = 'C:/Users/f1003/Desktop/hy/scalogram/' + leadFolder + '/Train'
    train_generator = data_generator.flow_from_directory(
            tr_dir,
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
    
    val_dir = 'C:/Users/f1003/Desktop/hy/scalogram/' + leadFolder + '/Val'
    validation_generator = data_generator.flow_from_directory(
            val_dir,
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
    
    
    
    ## model setup
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import InceptionResNetV2
#    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, GlobalMaxPool2D, GlobalAveragePooling2D
    from tensorflow.keras import optimizers
    from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    model = Sequential()
    
    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(InceptionResNetV2(include_top = False, pooling = InceptionResNetV2_POOLING_AVERAGE, weights = "imagenet"))
    
    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
    
    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = True
        
    # model.summary()
        
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)
    
    
    cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
    cb_checkpointer = ModelCheckpoint(filepath = 'C:/Users/f1003/Desktop/hy/scalogram/working/'+leadFolder+'/0.001_InceptionResNetV2_32.hdf5', 
                                      monitor = 'val_loss', 
                                      save_best_only = True, 
                                      mode = 'auto')
    
    
    fit_history = model.fit_generator(
            train_generator,
            steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
            epochs = NUM_EPOCHS,
            validation_data=validation_generator,
            validation_steps=STEPS_PER_EPOCH_VALIDATION,
            callbacks=[cb_checkpointer, cb_early_stopper]
    )
    
    model.load_weights("C:/Users/f1003/Desktop/hy/scalogram/working/"+leadFolder+"/0.001_InceptionResNetV2_32.hdf5")
    
    
    print(fit_history.history.keys())
    
    plt.figure(1, figsize = (15,8)) 
        
    plt.subplot(221)  
    plt.plot(fit_history.history['accuracy'])  
    plt.plot(fit_history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'valid']) 
        
    plt.subplot(222)  
    plt.plot(fit_history.history['loss'])  
    plt.plot(fit_history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'valid']) 
    
    plt.show()
    
    tst_dir = 'C:/Users/f1003/Desktop/hy/scalogram/' + leadFolder + '/Test'
    test_generator = data_generator.flow_from_directory(
        directory = tst_dir,
        target_size = (image_size, image_size),
        batch_size = BATCH_SIZE_TESTING,
        shuffle=False)
    
    test_generator.reset()
    
    y_test = test_generator.classes
    pred = model.predict_generator(test_generator)#, steps = len(test_generator), verbose = 1)
    
    y_test_pred = np.argmax(pred, axis = 1)
    
    # loss, acc = model.evaluate_generator(test_generator)
    
    ## performance measure
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.metrics import precision_score, roc_auc_score, confusion_matrix
    
    # y_test_prob = bestModel.predict_proba(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, pred, multi_class='ovr')
    rec_macro = recall_score(y_test, y_test_pred, average='macro')
    pre_macro = precision_score(y_test, y_test_pred, average='macro')
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    rec_micro = recall_score(y_test, y_test_pred, average='micro')
    pre_micro = precision_score(y_test, y_test_pred, average='micro')
    f1_micro = f1_score(y_test, y_test_pred, average='micro')
    rec_weighted = recall_score(y_test, y_test_pred, average='weighted')
    pre_weighted = precision_score(y_test, y_test_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    


    # save performance measures
    fname.append(leadFolder)
    accuracy.append(acc)
    AUCs.append(auc)
    recall_macro.append(rec_macro)
    precision_macro.append(pre_macro)
    f1score_macro.append(f1_macro)
    recall_micro.append(rec_micro)
    precision_micro.append(pre_micro)
    f1score_micro.append(f1_micro)
    recall_weighted.append(rec_weighted)
    precision_weighted.append(pre_weighted)
    f1score_weighted.append(f1_weighted)

    nameTmp = leadFolder + ' is done'
    print(nameTmp)


import pandas as pd

df = pd.DataFrame([accuracy, AUCs, recall_macro, precision_macro, f1score_macro, 
                   recall_micro, precision_micro, f1score_micro, 
                   recall_weighted, precision_weighted, f1score_weighted],
  index=['accuracy','AUC','recall_macro','precision_macro','f1 score_macro',
         'recall_micro','precision_micro','f1 score_micro',
         'recall_weighted','precision_weighted','f1 score_weighted'], columns=fname)


# write excel file
fileName = 'sc_results_InceptionResNetV2_lr'+str(lr)+'.xlsx'
df.to_excel('C:/Users/f1003/Desktop/hy/scalogram/ECG2grayImg_trValTst/'+fileName)


