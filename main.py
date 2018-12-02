from glob import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np

from time import time

from keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout, Activation, BatchNormalization, Add
from keras import Model

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

images = glob('UTKFace/*.jpg')
RaceLabels = ['white', 'black', 'asian', 'indian', 'others']
GenderLabels = ['male', 'female']
min_age = 20
max_age = 50

def getDataHistogram(images):

    ageD = {}
    genderD = {}
    raceD = {}

    for i, img in enumerate(images):

        img = img.split('/')[-1]
        img = img.split('.')[0]
        age, gender, race, date = img.split('_')
        age = int(age)
        gender = int(gender)
        race = int(race)

        if age>=min_age and age<=max_age:

            if age in ageD:
                ageD[age].append(i)
            else:
                ageD[age] = [i]

            if gender in genderD:
                genderD[gender].append(i)
            else:
                genderD[gender] = [i]

            if race in raceD:
                raceD[race].append(i)
            else:
                raceD[race] = [i]

    plt.figure('Age Histogram')
    x = sorted(ageD.keys())
    y = [len(ageD[k]) for k in x]
    plt.xlabel('Age')
    plt.ylabel('Count')
    for i,X in enumerate(x):
        plt.vlines(X, 0, y[i], colors='k', linestyles='solid')
    plt.plot(x,y, linestyle='None', marker='o')

    plt.figure('Gender Histogram')
    x = sorted(genderD.keys())
    y = [len(genderD[k]) for k in x]
    plt.xlabel('Gender')
    plt.ylabel('Count')
    x = GenderLabels
    for i,X in enumerate(x):
        plt.vlines(X, 0, y[i], colors='k', linestyles='solid')
    plt.plot(x,y, linestyle='None', marker='o')

    plt.figure('Race Histogram')
    x = sorted(raceD.keys())
    y = [len(raceD[k]) for k in x]
    plt.xlabel('Race')
    plt.ylabel('Count')
    x = RaceLabels
    for i,X in enumerate(x):
        plt.vlines(X, 0, y[i], colors='k', linestyles='solid')
    plt.plot(x,y, linestyle='None', marker='o')

    plt.show()

def getDataset(min_age, max_age):

    ages = {}
    races = {}

    dataset = []

    for img in tqdm(images):
        imgName = img.split('/')[-1]
        age, gender, race, _ = imgName.split('_')
        age = int(age)
        race = int(race)
        image = cv2.imread(img)

        if age >= min_age and age <= max_age:
            if age in ages:
                ages[age] += 1
            else:
                ages[age] = 1

            if race in races:
                races[race] += 1
            else:
                races[race] = 1
            dataset.append( (image, race) )

    # show age distribution
    x,y = [], []
    for a in sorted([int(ag) for ag in ages.keys()]):
        x.append(a)
        y.append(ages[a])
    plt.plot(x, y)
    plt.show()

    # show race distribution
    x, y = [], []
    for r in sorted([int(rc) for rc in races.keys()]):
        x.append(r)
        y.append(races[r])
    plt.plot(RaceLabels, y, linestyle='None', marker='o')
    plt.show()

    return dataset


def getModel(version = 1):

    if version==1: # Baseline

        inp = Input(shape=(200, 200, 3,))

        net = Conv2D(filters=16, strides=(2,2), kernel_size=(3,3))(inp)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=32, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Flatten()(net)

        net = Dense(512, activation='relu')(net)
        out = Dense(5, activation='softmax')(net)

        model = Model(inputs=[inp], outputs=[out])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model

    elif version==2: # Mininet

        inp = Input(shape=(200, 200, 3,))

        net = Conv2D(filters=16, strides=(2,2), kernel_size=(3,3))(inp)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=32, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Flatten()(net)

        net = Dense(128, activation='relu')(net)
        out = Dense(5, activation='softmax')(net)

        model = Model(inputs=[inp], outputs=[out])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model

    elif version==3: # Widenet

        inp = Input(shape=(200, 200, 3,))

        net = Conv2D(filters=32, strides=(2,2), kernel_size=(3,3))(inp)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Flatten()(net)

        net = Dense(512, activation='relu')(net)
        out = Dense(5, activation='softmax')(net)

        model = Model(inputs=[inp], outputs=[out])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model

    elif version==4: #Hugenet

        inp = Input(shape=(200, 200, 3,))

        net = Conv2D(filters=32, strides=(2,2), kernel_size=(3,3))(inp)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=256, kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Conv2D(filters=256, kernel_size=(3,3))(net)
        net = BatchNormalization()(net)
        net = Activation('elu')(net)
        net = Dropout(0.5)(net)

        net = Flatten()(net)

        net = Dense(256, activation='relu')(net)
        net = Dense(512, activation='relu')(net)
        out = Dense(5, activation='softmax')(net)

        model = Model(inputs=[inp], outputs=[out])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model

def getOneHot(Y, shouldOneHot=True):
    Y_onehot = np.zeros((len(Y), 5))

    for i,y in enumerate(Y):
        if shouldOneHot:
            Y_onehot[i, y] = 1
        else:
            Y_onehot[i] = y

    Y = Y_onehot
    return Y


def getData(min_age, max_age):

    datafile = str(min_age) + '_' + str(max_age) + '_dataset.pkl'
    dataset = pickle.load(open(datafile, 'rb'))

    X = np.array([i for i,j in dataset])
    Y = np.array([j for i,j in dataset])

    return X, Y


def evaluate(p, y):

    acc = 0

    P_lab = np.argmax(p)
    Y_lab = np.argmax(y)

    for i in range(len(p)):
        if np.argmax(p[i]) == np.argmax(y[i]):
            acc += 1

    return acc / len(p)


def KFoldTrain():

    K = 5

    X, Y = getData(min_age, max_age)
    models = [getModel(version=2) for _ in range(K)]

    skf = StratifiedKFold(n_splits=K)
    skf.get_n_splits(X,Y)

    accuracies = []
    epochs = 20

    for e in range(epochs):

        print('epoch', e+1)
        A = []

        for i, (train_index, test_index) in tqdm(list(enumerate(skf.split(X, Y)))):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = getOneHot(Y)[train_index], getOneHot(Y)[test_index]

            print('\nfold ' + str(i))
            models[i].fit(X_train, y_train, batch_size=64, validation_split=0.1)
            # acc = models[i].evaluate(X_test, y_test)
            preds = models[i].predict(X_test)
            acc = evaluate(preds, y_test)
            print('test accuracy', acc)
            A.append(acc)

        mA = np.mean(A)
        sA = 0
        # sA = np.std(A)

        print("EPOCH", e, "accuracies:", mA, '+/-', sA)

        accuracies.append( (mA, sA) )

        print('Accuracies over epochs', accuracies)


def saveTrainTestData(X, Y, split):

    skf = StratifiedKFold(n_splits=int(1/split))
    skf.get_n_splits(X,Y)
    tr, te = list(skf.split(X, Y))[0]

    Xtrain, Xtest, Ytrain, Ytest = X[tr], X[te], Y[tr], Y[te]
    pickle.dump((Xtrain, Xtest, Ytrain, Ytest), open('train_test_dataset.pkl', 'wb'))


def loadTrainTestData():

    Xtr, Xte, Ytr, Yte = pickle.load(open('train_test_dataset.pkl', 'rb'))
    # print(Ytr, Yte)
    return np.array(Xtr), np.array(Xte), np.int32(Ytr), np.int32(Yte)



# print('loading images, minage', min_age, 'maxage', max_age)
# dataset = getDataset(min_age, max_age)

# print('writing to pickle file' + str(min_age) + '_' + str(max_age) + '_dataset.pkl')
# pickle.dump(dataset, open(str(min_age) + '_' + str(max_age) + '_dataset.pkl', 'wb'))

# X,Y = getData(min_age, max_age)
# saveTrainTestData(X, Y, split=0.1)

def train(run=1, version=1, epochs=30):

    Xtr, Xte, Ytr, Yte = loadTrainTestData()
    Ytr, Yte = getOneHot(Ytr), getOneHot(Yte)

    model = getModel(version=version)

    trloss = []
    tracc = []
    teloss = []
    teacc = []
    trainingTimes = []
    testingTimes = []

    for e in range(epochs):

        t1 = time()
        history = model.fit(Xtr, Ytr, batch_size=256)
        t2 = time()
        model.save_weights(str(run) + '_' + str(e) + '.h5')
        print('train metrics', history.history)

        t = (t2-t1) / Ytr.shape[0]
        print('training time per example:', t)
        trainingTimes.append(t)

        t3 = time()
        testhistory = model.evaluate(x=Xte, y=Yte)
        t4 = time()
        print('test metrics', testhistory)

        t = (t4-t3) / Yte.shape[0]
        print('inference time per example:', t)
        testingTimes.append(t)

        trloss.append(history.history['loss'][0])
        tracc.append(history.history['acc'][0])
        teloss.append(testhistory[0])
        teacc.append(testhistory[1])

        print("EPOCH", e, 'test loss', testhistory[0], 'test accuracy:', testhistory[1])

        print('trloss =', trloss)
        print('tracc =', tracc)
        print('teloss =', teloss)
        print('teacc =', teacc)
        print('trainingTimes =', trainingTimes)
        print('testingTimes =', testingTimes)

# KFoldTrain()
# train(run=1, version=1)
print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$ VERSION 1 $$$$$$$$$$$$$$$$$$$$$$$$\n\n')
train(run=1, version=1)
print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$ VERSION 2 $$$$$$$$$$$$$$$$$$$$$$$$\n\n')
train(run=2, version=2)
print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$ VERSION 3 $$$$$$$$$$$$$$$$$$$$$$$$\n\n')
train(run=3, version=3)
print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$ VERSION 4 $$$$$$$$$$$$$$$$$$$$$$$$\n\n')
train(run=4, version=4)

# getDataHistogram(images)
