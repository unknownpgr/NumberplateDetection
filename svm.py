import os
import cv2
import numpy as np

data = []
labels = []

binCount = 100


def getDescribeVector(img):
    size = img.shape[:2]
    w = size[1]/size[0]
    meta = np.array([size[0], size[1], w])

    # calculate grayscale image histogram
    img = cv2.resize(img, (520, 110))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    hist = cv2.calcHist([img], [0], None, [binCount], [0, 256])
    hist = np.resize(hist, [1, binCount])[0]

    # concatenate data
    return np.concatenate((hist, meta))


def getData(dirPath, label):
    for i in os.listdir(dirPath):
        # load image
        img = cv2.imread(dirPath+'/'+i)
        # if img == None:
        #     continue
        row = getDescribeVector(img)

        data.append(row)
        labels.append(label)


getData('./true', 1)
getData('./false', 0)

data = np.array(data, np.float32)
labels = np.array(labels, np.int)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(1.25)
svm.setGamma(100)
svm.train(data, cv2.ml.ROW_SAMPLE, labels)

testResponse = svm.predict(data)
print(testResponse)

svm.save('./svm_data.yml')
