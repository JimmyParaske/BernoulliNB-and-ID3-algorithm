import tensorflow as tf

from sklearn.metrics import classification_report
from tabulate import tabulate
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=4000)
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


def Train_Bayes(vocabulary, x_train_binary):
    count = 0
    Frequency = [[0 for i in range(len(vocabulary))] for x in range(2)]

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
            Likelihood[c][i] = 100 * Frequency[c][i] / Frequency_sum[c]
    return Likelihood


def Apply_Bayes(Likelihood, x_test_binary, y_test):
    count = 0
    accuracy = 0
    accuracy_elbow = [0 * i for i in range(25000)]
    acc_report = [0 * i for i in range(25000)]
    tp = [0 * i for i in range(25000)]
    fp = [0 * i for i in range(25000)]
    fn = [0 * i for i in range(25000)]
    tp_count = 0
    fp_count = 0
    fn_count = 0

    for review in x_test_binary:
        P_pos = 12500
        P_neg = 12500
        i = 0
        for r in review:
            if r == 1:
                P_pos = P_pos * Likelihood[0][i]
                P_neg = P_neg * Likelihood[1][i]
            # elif r==0:
            #   P_pos=P_pos*(1-Likelihood[0][i])
            #   P_neg=P_neg*(1-Likelihood[1][i])
            i += 1

        if ((P_pos > P_neg) and (y_test[count] == 0)) or ((P_pos < P_neg) and (y_test[count] == 1)):
            accuracy += 1

            if count > 0:
                tp_count += 1
                tp[count] = tp_count
            else:
                tp[0] += 1
        elif (P_pos > P_neg) and (y_test[count] == 1):

            if count > 0:
                fp_count += 1
                fp[count] = fp_count
            else:
                fp[0] += 1
        elif (P_pos < P_neg) and (y_test[count] == 0):

            if count > 0:
                fn_count += 1
                fn[count] = fn_count
            else:
                fn[0] += 1

        if P_neg > P_pos:
            acc_report[count] = 1

        accuracy_elbow[count] = (100 * accuracy + 1) / (count + 2)

        count += 1

    return [acc_report, accuracy_elbow, tp, fp, fn]


# pinakas = [i+1 for i in range(25000)]
# plt.xlabel("files")
# plt.ylabel("accuracy")
# plt.plot(pinakas, Apply_Bayes(Train_Bayes(vocabulary, x_train_binary), x_test_binary)[1])
# plt.show()

clasficdic = classification_report(y_test, Apply_Bayes(Train_Bayes(vocabulary, x_train_binary), x_test_binary, y_test)[0])
clasficdictable=clasficdic.split()
clatb=[[clasficdictable[0],clasficdictable[5]],[clasficdictable[1],clasficdictable[6]],[clasficdictable[2],clasficdictable[7]]]
col_names=["Results","Values"]
print(tabulate(clatb, headers=col_names, tablefmt="fancy_grid"))
#Apply_Bayes(Train_Bayes(vocabulary, x_train_binary), x_train_binary, y_train)

tp=Apply_Bayes(Train_Bayes(vocabulary, x_train_binary), x_train_binary, y_train)[2]
fp = Apply_Bayes(Train_Bayes(vocabulary, x_train_binary), x_train_binary, y_train)[3]
fn = Apply_Bayes(Train_Bayes(vocabulary, x_train_binary), x_train_binary, y_train)[4]
precision = [0 * i for i in range(25000)]
recall = [0 * i for i in range(25000)]
f1 = [0 * i for i in range(25000)]
for i in range(25000):
    precision[i] = (tp[i]+1) / (tp[i]+fp[i]+2)
    recall[i] = (tp[i]+1) / (tp[i]+fn[i]+2)
    f1[i] = (2*precision[i]*recall[i]) / (precision[i] + recall[i])


#pinakas = [i+1 for i in range(25000)]
pinakas = [2500*i for i in range(1,10)]
precision = [precision[i] for i in range(1,10)]
recall = [recall[i] for i in range(1,10)]
f1 = [f1[i] for i in range(1,10)]
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