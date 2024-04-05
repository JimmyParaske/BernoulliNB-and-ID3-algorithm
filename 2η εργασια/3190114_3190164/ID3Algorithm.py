import math
from turtle import right
from xml.etree.ElementTree import tostring
import tensorflow as tf

from sklearn.metrics import classification_report
from tabulate import tabulate
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=100)
word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i, word) for (word, i) in word_index.items())
decoded_sequence = list(" ".join(index2word[i] for i in text) for text in x_train)
vocabulary = list()
for text in decoded_sequence:
    words = text.split()
    for word in words:
        if not (word in vocabulary):
            vocabulary.append(word)

x_train_binary = list()
x_test_binary = list()
for text in decoded_sequence:
    tokens = text.split()
    binary_vector = list()
    for vocab_token in vocabulary:
        if vocab_token in tokens:
            binary_vector.append(1)
        else:
            binary_vector.append(0)

    x_train_binary.append(binary_vector)

decoded_sequence_test = list(" ".join(index2word[i] for i in text) for text in x_test)
for text in decoded_sequence_test:
    tokens = text.split()
    binary_vector = list()
    for vocab_token in vocabulary:
        if vocab_token in tokens:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    x_test_binary.append(binary_vector)


def twoEntropy(cProb):
    if cProb == 0 or cProb == 1:
        return 0.0
    else:
        return -(cProb * math.log2(cProb)) - ((1.0 - cProb) * math.log2(1.0 - cProb))


def calcIG(table, vocabulary, y_train):
    numOfExamples = len(table)
    numOfFeatures = len(vocabulary)

    IG = [0 * i for i in range(numOfFeatures)]

    positives = 0
    for i in y_train:
        if i == 1:
            positives += 1

    PC1 = positives / numOfExamples

    HC = twoEntropy(PC1)

    PX1 = [0 * i for i in range(numOfFeatures)]

    PC1X1 = [0 * i for i in range(numOfFeatures)]

    PC1XO = [0 * i for i in range(numOfFeatures)]

    HCX1 = [0 * i for i in range(numOfFeatures)]

    HCX0 = [0 * i for i in range(numOfFeatures)]

    for j in range(numOfFeatures):

        cX1 = 0
        cC1X1 = 0
        cC1X0 = 0

        for i in range(numOfExamples):
            if table[i][j] == 1:
                cX1 += 1
            if table[i][j] == 1 and y_train[i] == 1:
                cC1X1 += 1
            if table[i][j] == 0 and y_train[i] == 1:
                cC1X0 += 1

        PX1[j] = cX1 / numOfExamples

        if cX1 == 0:
            PC1X1[j] = 0.0
        else:
            PC1X1[j] = cC1X1 / cX1

        if cX1 == numOfExamples:
            PC1XO[j] = 0.0
        else:
            PC1XO[j] = cC1X0 / (numOfExamples - cX1)

        HCX1[j] = twoEntropy(PC1X1[j])
        HCX0[j] = twoEntropy(PC1XO[j])

        IG[j] = HC - ((PX1[j] * HCX1[j]) + ((1.0 - PX1[j]) * HCX0[j]))

    return IG


def idiaKathgoria(y_train):
    c = y_train[0]
    count = 0
    for i in y_train:
        if i == c:
            count += 1

    if count == len(y_train):
        return True
    else:
        return False


def syxnoterhKathgoria(y_train):
    countPos = 0
    countNeg = 0
    for i in y_train:
        if y_train[i] == 1:
            countPos += 1
        else:
            countNeg += 1

    if max(countPos, countNeg) == countPos:
        return 1
    else:
        return 0


class TreeForID3:

    def __init__(self, best):
        self.question = vocabulary[best]

        self.left = None
        self.right = None


def diafora(vocabulary, best):
    vocabulary.pop(best)

    return vocabulary


def ID3Train(x_train_binary, y_train, vocabulary, proepilegmenh):
    if not x_train_binary:
        return proepilegmenh
    elif idiaKathgoria(y_train):
        return y_train[0]
    elif not vocabulary:
        return syxnoterhKathgoria(y_train)
    else:
        igtable = calcIG(x_train_binary, vocabulary, y_train)
        best = igtable.index(max(igtable))
        tree = TreeForID3(best)

        m = syxnoterhKathgoria(y_train)
        tableExist = []
        tableNotExist = []
        labelExist = []
        labelNotExist = []

        for i in x_train_binary:
            if i[best] == 1:
                tableExist.append(i)
                labelExist.append(y_train[x_train_binary.index(i)])
            else:
                tableNotExist.append(i)
                labelNotExist.append(y_train[x_train_binary.index(i)])

        tree.left = ID3Train(tableExist, labelExist, vocabulary, m)
        tree.right = ID3Train(tableNotExist, labelNotExist, vocabulary, m)
        return tree


def ID3Test(x_test_binary, y_test, vocabulary, dentro):
    count = 1
    tp_count = 0
    fn_count = 0
    fp_count = 0
    tp = [0 * i for i in range(25000)]
    fp = [0 * i for i in range(25000)]
    fn = [0 * i for i in range(25000)]
    accuracy = 0
    for i in x_test_binary:
        newTree = dentro
        while type(newTree) == TreeForID3:
            if i[vocabulary.index(newTree.question)] == 1:
                newTree = newTree.left
            else:
                newTree = newTree.right
        if newTree == y_test[count - 1]:
            accuracy += 1
            if newTree == 1:
                tp_count += 1
                tp[count - 1] = tp_count
        else:
            if newTree == 1:
                fp_count += 1
                fp[count - 1] = fp_count
            else:
                fn_count = 1
                fn[count - 1] = fn_count
        count += 1
    return [accuracy / 25000, tp, fp, fn]


vocabulary_copy = vocabulary
dentro = ID3Train(x_train_binary, y_train, vocabulary_copy, 0)
id3 = ID3Test(x_test_binary, y_test, vocabulary, dentro)
tp = id3[1]
fp = id3[2]
fn = id3[3]

precision = [0 * i for i in range(25000)]
recall = [0 * i for i in range(25000)]
f1 = [0 * i for i in range(25000)]

for i in range(25000):
    precision[i] = (tp[i] + 1) / (tp[i] + fp[i] + 2)
    recall[i] = (tp[i] + 1) / (tp[i] + fn[i] + 2)
    f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])

precision = [precision[i] for i in range(0, 25000, 999)]
recall = [recall[i] for i in range(0, 25000, 999)]
f1 = [f1[i] for i in range(0, 25000, 999)]
pinakas = [i for i in range(0, 25000, 999)]

plt.xlabel("files")
plt.ylabel("precision")
plt.plot(pinakas, precision)
plt.show()

plt.xlabel("files")
plt.ylabel("recall")
plt.plot(pinakas, recall)
plt.show()

plt.xlabel("files")
plt.ylabel("f1")
plt.plot(pinakas, f1)
plt.show()

new_pinakas = [i * 1000 for i in range(25)]
acc_train = []
acc_test = []
for i in new_pinakas:
    vocabulary_copy = vocabulary
    proepilegmenh = 0
    new_x_train_binary = []
    new_y_train_binary = []
    if i == 0:
        new_x_train_binary.append(x_train_binary[0])
        new_y_train_binary.append(y_train[0])
    else:
        for z in range(i):
            new_x_train_binary.append(x_train_binary[z])
            new_y_train_binary.append(y_train[z])
    if i % 500 == 0:
        print(i)
    dentro = ID3Train(new_x_train_binary, new_y_train_binary, vocabulary_copy, proepilegmenh)
    acc_train.append(ID3Test(x_train_binary, y_train, vocabulary, dentro)[0])
    acc_test.append(ID3Test(x_test_binary, y_test, vocabulary, dentro)[0])

clatb = [["precision", precision[-1]], ["recall", recall[-1]],
         ["f1", f1[-1]]]



col_names = ["Results", "Values"]
print(tabulate(clatb, headers=col_names, tablefmt="fancy_grid"))

clatb = []

clatb1 = []

for i in range(25):
    print(new_pinakas[i])
    clatb.append([new_pinakas[i], acc_train[i]])
    clatb1.append([new_pinakas[i], acc_test[i]])


print(tabulate(clatb, headers=col_names, tablefmt="fancy_grid"))
print(tabulate(clatb1, headers=col_names, tablefmt="fancy_grid"))