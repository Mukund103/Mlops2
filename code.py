from keras.datasets import mnist

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

from keras.optimizers import SGD

(trainX,trainY),(testX,testY)=mnist.load_data()

def data_prep(trainX,trainY,testX,testY):
  trainX=trainX.reshape((trainX.shape[0],28,28,1))
  testX=testX.reshape((testX.shape[0],28,28,1))
  trainX=trainX.astype('float32')
  testX=testX.astype('float32')
  trainX=trainX/255.0
  testX=testX/255.0
  trainY=to_categorical(trainY)
  testY=to_categorical(testY)
  return trainX,trainY,testX,testY

trainX,trainY,testX,testY=data_prep(trainX,trainY,testX,testY)
def model():
  model=Sequential()
  model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(units=100,activation='relu'))
  model.add(Dense(units=10,activation="softmax",))
  model.compile(optimizer=SGD(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
  return model
def prep_pixels(train,test):
  train_norm=train.astype('float32')
  test_norm=test.astype('float32')
  train_norm=train_norm/255.0
  test_norm=test_norm/255.0
  return train_norm,test_norm

trainX,testX=prep_pixels(trainX,testX)

Model=model()

history=Model.fit(trainX,trainY,epochs=3)

Model.save("/Code/mnist.h5")








