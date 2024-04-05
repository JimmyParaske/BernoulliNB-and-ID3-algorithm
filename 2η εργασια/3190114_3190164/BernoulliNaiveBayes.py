import tensorflow as tf

from sklearn.metrics import classification_report
from tabulate import tabulate
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=500)
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


def Train_Bayes(vocabulary, x_train_binary, y_train):
    count = 0
    Frequency = [[0 * i for i in range(len(vocabulary))] for x in range(2)]

    for review in x_train_binary:
        i = 0
        for word in review:
            Frequency[y_train[count]][i] += word
            i += 1
        count += 1
    Frequency_sum = [0, 0]

    for c in range(2):
        Frequency_sum[c] = sum(Frequency[c])
    Likelihood = [[0 for i in range(len(vocabulary))] for x in range(2)]

    for i in range(len(vocabulary)):
        for c in range(2):
            Likelihood[c][i] = (Frequency[c][i] + 1) / (Frequency_sum[c] + len(vocabulary))
    return Likelihood


def Apply_Bayes(Likelihood, x_test_binary, y_test):
    count = 0
    accuracy = 0
    # accuracy_elbow = [0 * i for i in range(25000)]
    acc_report = [0 * i for i in range(25000)]
    tp = [0 * i for i in range(25000)]
    fp = [0 * i for i in range(25000)]
    fn = [0 * i for i in range(25000)]
    tp_count = 0
    fp_count = 0
    fn_count = 0

    for review in x_test_binary:
        P_pos = 0.5
        P_neg = 0.5
        i = 0
        for r in review:
            if r == 1:
                P_pos = P_pos * Likelihood[1][i]
                P_neg = P_neg * Likelihood[0][i]
            i += 1

        if ((P_pos > P_neg) and (y_test[count] == 1)) or ((P_pos < P_neg) and (y_test[count] == 0)):
            accuracy += 1

        if ((P_pos > P_neg) and (y_test[count] == 1)):
            tp_count += 1
            tp[count] = tp_count
        elif (P_pos > P_neg) and (y_test[count] == 0):
            fp_count += 1
            fp[count] = fp_count

        elif (P_pos < P_neg) and (y_test[count] == 1):
            fn_count += 1
            fn[count] = fn_count

        if P_neg > P_pos:
            acc_report[count] = 1

        # accuracy_elbow[count] = (accuracy + 1) / (count + 2)

        count += 1
    true_acc = accuracy / count
    return [acc_report, true_acc, tp, fp, fn]


Bayes = Apply_Bayes(Train_Bayes(vocabulary, x_train_binary, y_train), x_train_binary, y_train)
clasficdic = classification_report(y_test, Bayes[0])
clasficdictable = clasficdic.split()
clatb = [[clasficdictable[0], clasficdictable[5]], [clasficdictable[1], clasficdictable[6]],
         [clasficdictable[2], clasficdictable[7]]]
col_names = ["Results", "Values"]
print(tabulate(clatb, headers=col_names, tablefmt="fancy_grid"))

tp = Bayes[2]
fp = Bayes[3]
fn = Bayes[4]

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
new_pinakas = [i*100 + 100 for i in range(250)]
acc_train=[]
acc_test=[]
for i in new_pinakas:
    new_x_train_binary=[]
    new_y_train_binary=[]
    if i==0:
        new_x_train_binary.append(x_train_binary[0])
        new_y_train_binary.append(y_train[0])
    else:
        for z in range(i):
            new_x_train_binary.append(x_train_binary[z])
            new_y_train_binary.append(y_train[z])
    if i%500==0:
        print(i)
    Bayes=Train_Bayes(vocabulary,new_x_train_binary,new_y_train_binary)
    acc_train.append(Apply_Bayes(Bayes,x_train_binary,y_train)[1])
    acc_test.append(Apply_Bayes(Bayes,x_test_binary,y_test)[1])


plt.xlabel("train files")
plt.ylabel("accuracy")
plt.plot(new_pinakas, acc_train)
plt.plot(new_pinakas, acc_test)
plt.show()


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

clatb = []
clatb1 = []



for i in range(250):
    clatb.append([new_pinakas[i], acc_train[i]])
    clatb1.append([new_pinakas[i], acc_test[i]])

col_names = ["Results", "Values"]
print(tabulate(clatb, headers=col_names, tablefmt="fancy_grid"))
print(tabulate(clatb1, headers=col_names, tablefmt="fancy_grid"))