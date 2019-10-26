## image process program, transfer color image to 28 * 28 black and write image
## From https://www.jianshu.com/p/82387ae42587
import os
from skimage import io
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# image size
N = 28
# gray level threshold
color = 100/255
# Letters lookup
letterstable = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

def GetTrainPicture (files):
    # Picture = np.zeros([len(files), N ** 2 + 1])
    # for i, item in enumerate(files):
    img = io.imread('./num/' + files[0], as_gray = True)

    img[img > color] = 1
    img = cv2.blur(img, (3, 3))
    io.imshow(img)
    plt.show()
    ret,img=cv2.threshold(img, 0.6, 1, cv2.THRESH_BINARY_INV)
    io.imshow(img)
    plt.show()

    img = CutPicture(img)
    # k = np.ones((5, 5), np.uint8)
    # img =cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=2)
    io.imshow(img)
    plt.show()
    # img  = cv2.resize(img , (300 if len(img[0]) > 300 else len(img[0]), 100 if len(img) > 100 else len(img)))
    (img, maxlabel) = Rangegrow(img)
    # print('max' + str(maxlabel))
    io.imshow(img)
    plt.show()
    letters = []
    for i in range(maxlabel - 2):
        letters.append(np.zeros((len(img), len(img[0]))))
    Getletters(img, letters)
    # print('len' + str(len(letters)))
    for i in range(len(letters)):
        letters[i] = CutPicture(letters[i])
        # io.imshow(img)
        # plt.show()
        # print('size' + str(letters[i].size))
        # io.imshow(letters[i])
        # plt.show()
        if letters[i].size >= 10:
            # k = np.ones((3, 3), np.uint8)
            # letters[i] = cv2.dilate(letters[i], k, iterations=2)
            letters[i]  = cv2.resize(letters[i] , (N - 2, N - 2))
            letters[i][letters[i] >= 0.7] = 1
            letters[i][letters[i] < 0.7] = 0
            letters[i] = Addframe(letters[i])
    
            letters[i] = letters[i].reshape(1,N * N)
            letters[i] = letters[i].reshape(1, N, N, 1)

    # Picture[i, :N ** 2] = img.reshape(1, N * N)
    # Picture[Picture > 0.3] = 1
    # Picture[Picture <= 0.3] = 0
    # Picture[i, N ** 2] = 1
    return letters
def Addframe(img):
    new_pic = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, value=0)
    return new_pic

def Getletters(img,letters):
    # print(img.shape, len(letters))
    for y in range(len(img[0])):
        for x in range(len(img)):
            for i in range(2, len(letters) + 2):
                if img[x][y] == i:
                    letters[i - 2][x][y] = 1

def Rangegrow(img):
    # Seperate letters between each other
    img2 = np.zeros((len(img), len(img[0]) ))
    current_label = 2
    for y in range(1, len(img[0]) - 1):
        for x in range(1, len(img) - 1):
            if img[x][y] == 1:
                if img2[x][y] == 0:
                    setlabel(img2, img, x, y, current_label)
                    current_label += 1
    return img2, current_label

def setlabel(img2, img, x, y, n):
    img2[x][y] = n
    if x > 0 and y > 0 and x < len(img) - 1 and y < len(img[0]) - 1:
        if img[x][y - 1] == 1 and img2[x][y - 1] == 0:
            setlabel(img2, img, x, y - 1, n)
        if img[x - 1][y] == 1 and img2[x - 1][y] == 0:
            setlabel(img2, img, x - 1, y, n)
        if img[x][y + 1] == 1 and img2[x][y + 1] == 0:
            setlabel(img2, img, x, y + 1, n)
        if img[x + 1][y] == 1 and img2[x + 1][y] == 0:
            setlabel(img2, img, x + 1, y, n)


def CutPicture(img):
    size = []
    l = len(img)
    w = len(img[0, :])

    size.append(JudgeEdge(img, l, 0, [-1, -1]))
    size.append(JudgeEdge(img, w, 1, [-1, -1]))
    size = np.array(size).reshape(4)
    # four values in size: the highest row index of the letter
    #                      the lowest row index of the letter
    #                      the leftest column index of the letter
    #                      the rightest column index fo the letter
    return img[size[0] : size[1] + 1, size[2]: size[3] + 1]

def JudgeEdge(img, length, flag, size):
## flag = 0  length; flag = 1 width
    for i in range(length):
        if not flag:
            line1 = img[i, img[i, :] > 0]
            line2 = img[length - 1 - i, img[length - 1 - i, :] > 0]
        else:
            line1 = img[img[:, i] > 0, i]
            line2 = img[img[:, length - 1 - i] > 0, length - 1 - i]
        if len(line1) >= 1 and size[0] == -1:
            size[0] = i
        if len(line2) >= 1 and size[1] == -1:
            size[1] = length - 1 - i
        if size[0] != -1 and size[1] != -1:
            break
    return size


cnn_model = load_model('emnist_cnn_model.h5')
filenames = os.listdir(r"./num/")
letters = GetTrainPicture(filenames)
for letter in letters:
    # print(letter.size)
    if letter.size == N * N:
        prediction1 = cnn_model.predict(letter)[0]
        prediction1 = np.argmax(prediction1)
        print(letterstable[prediction1 + 1])