from keras.utils import multi_gpu_model
from keras.optimizers import Adamax
from generator import *
from EINet import *

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1' 

train_batch_size = 32
valid_batch_size = 5
save_interval = 1

epochs = 400

TRAIN_NOPIXEL_DIR = './data/TRN/Nopixel/'
TRAIN_WITHPIXEL_DIR = './data/TRN/S4pixel/'

VALID_NOPIXEL_DIR = './data/VAL/Nopixel/'
VALID_STEGO_DIR = './data/VAL/S4pixel/'

train_num = len(glob(TRAIN_NOPIXEL_DIR + '/*'))
valid_num = len(glob(VALID_NOPIXEL_DIR + '/*'))

train_gen = gen_train(TRAIN_NOPIXEL_DIR, TRAIN_WITHPIXEL_DIR, train_batch_size)
valid_gen = gen_valid(VALID_NOPIXEL_DIR, VALID_WITHPIXEL_DIR, valid_batch_size)

log_path = './weights/S4pixel_0/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
pretrained_weights = None

base_model = einet(input_size=(256, 256, 1))
base_model.summary()

model = multi_gpu_model(base_model, gpus=2)
optimizer = Adamax(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

if pretrained_weights:
    base_model.load_weights(pretrained_weights, by_name=True)

checkpointer = ParallelModelCheckpoint(base_model, filepath=log_path+'best_model.h5',
                               monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True,
                               mode='auto', period=save_interval)

ShowLog = ShowLR(log_path=log_path)
history = model.fit_generator(train_gen, steps_per_epoch=train_num//train_batch_size, epochs=epochs, verbose=2,
                              callbacks=[checkpointer, ShowLog], validation_data=valid_gen,
                              validation_steps=valid_num//valid_batch_size, max_queue_size=16, workers=1,
                              use_multiprocessing=False, initial_epoch=0)
with open(log_path+'Loss_Acc_Val_Loss_Val_Acc.txt', 'w') as f:
    f.write(str(history.history))
base_model.save_weights(log_path+'last_model.h5')

epochs = 200

TRAIN_NOPIXEL_DIR = './data/TRN/Nopixel/'
TRAIN_WITHPIXEL_DIR = './data/TRN/S4pixel/'

VALID_NOPIXEL_DIR = './data/VAL/Nopixel/'
VALID_STEGO_DIR = './data/VAL/S4pixel/'

train_num = len(glob(TRAIN_NOPIXEL_DIR + '/*'))
valid_num = len(glob(VALID_NOPIXEL_DIR + '/*'))

train_gen = gen_train(TRAIN_NOPIXEL_DIR, TRAIN_WITHPIXEL_DIR, train_batch_size)
valid_gen = gen_valid(VALID_NOPIXEL_DIR, VALID_STEGO_DIR, valid_batch_size)

log_path = './weights/S4pixel_1/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
pretrained_weights = './weights/Resized/S4pixel_0/best_model.h5'

base_model = einet(input_size=(256, 256, 1), drop_rate=0.5)
base_model.summary()

model = multi_gpu_model(base_model, gpus=2)
optimizer = Adamax(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

if pretrained_weights:
    base_model.load_weights(pretrained_weights, by_name=True)

checkpointer = ParallelModelCheckpoint(base_model, filepath=log_path+'best_model.h5',
                               monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True,
                               mode='auto', period=save_interval)

ShowLog = ShowLR(log_path=log_path)
history = model.fit_generator(train_gen, steps_per_epoch=train_num//train_batch_size, epochs=epochs, verbose=2,
                              callbacks=[checkpointer, ShowLog], validation_data=valid_gen,
                              validation_steps=valid_num//valid_batch_size, max_queue_size=16, workers=1,
                              use_multiprocessing=False, initial_epoch=0)
with open(log_path+'Loss_Acc_Val_Loss_Val_Acc.txt', 'w') as f:
    f.write(str(history.history))

####################################################################################
####################################################################################
test_batch_size = 50

TEST_NOPIXEL_DIR = './data/TST/Nopixel/'
TEST_S4pixel_DIR = './data/TST/S4pixel/'

test_num = len(glob(TEST_NOPIXEL_DIR + '/*'))

test_gen = gen_valid(TEST_NOPIXEL_DIR, TEST_S4pixel_DIR, test_batch_size)

pretrained_weights = './weights/S4pixel_1/best_model.h5'
if pretrained_weights:
    base_model.load_weights(pretrained_weights, by_name=True)

shot_num = 0
for i in range(0, test_num, test_batch_size):
    x_test, y_test = next(test_gen)
    y_pred = model.predict(x_test)
    y_pred_label = y_pred.argmax(axis=1)
    shot_num += np.sum(np.equal(y_pred_label, y_test[:, 1]))
print('testing accuracy: %f' % (shot_num/(test_num*2)))
####################################################################################
####################################################################################
