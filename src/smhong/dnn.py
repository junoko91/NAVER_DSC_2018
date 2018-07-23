import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# read csv
data = pd.read_csv("../../data/smhong/data.csv")

# data shufflinh
data = shuffle(data)

# data info 출력
data.info()

# Input / Output Data 분류
raw_Y = data["Position"]
raw_X = data.drop("Position", axis=1)

mdict = dict(enumerate(raw_Y.unique()))
raw_Y.replace(to_replace=mdict.values(), value=mdict.keys(), inplace=True)

print(raw_Y.values)
# Training Data / Test Data 분류
X_train, X_test, Y_train, Y_test = train_test_split(raw_X.values, raw_Y.values, test_size=0.3, random_state=42)

print("X_Train Shape :", np.shape(X_train))
print("Y_Train Shape :", np.shape(Y_train))
print(Y_train.ravel())

y_test = np.reshape(np.asarray(Y_test), [len(Y_test), 1])
y_train = np.reshape(np.asarray(Y_train.ravel()), [len(Y_train), 1])


#print("Classification Classes :", raw_Y.unique())

nbclass = len(raw_Y.unique())
nbfeature = np.shape(X_test)[1]

X = tf.placeholder(tf.float32, [None, nbfeature])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nbclass)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nbclass])

print(np.shape(Y_one_hot))

size_l1 = 256
size_l2 = size_l1 / 2
size_l3 = size_l2 / 2
size_l4 = size_l3 / 2

l1 = tf.layers.dense(X, size_l1, activation=tf.nn.relu, bias_initializer=tf.contrib.layers.xavier_initializer())
# l1 = tf.layers.batch_normalization(l1)
l2 = tf.layers.dense(l1, size_l2, activation=tf.nn.relu, bias_initializer=tf.contrib.layers.xavier_initializer())
# l2 = tf.layers.batch_normalization(l2)
l3 = tf.layers.dense(l2, size_l3, activation=tf.nn.relu, bias_initializer=tf.contrib.layers.xavier_initializer())
# l3 = tf.layers.batch_normalization(l3)
l4 = tf.layers.dense(l3, size_l4, activation=tf.nn.relu, bias_initializer=tf.contrib.layers.xavier_initializer())
# l4 = tf.layers.batch_normalization(l4)
l5 = tf.layers.dense(l1, nbclass, activation=None)

hypothesis = tf.nn.softmax(l5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l5, labels=Y_one_hot))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

correct_pred = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("train start")

sess = tf.Session()

tf.summary.merge_all()
train_writer = tf.summary.FileWriter("tensorboard/train", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

epoch = 200000
batch_size = 1
mok = int(len(X_train) / batch_size)

repeat = 0
index = 1
for i in range(epoch * mok):
    if index <= mok:
        x_train_batch = X_train[(index-1) * batch_size: index * batch_size, :]
        y_train_batch = y_train[(index-1) * batch_size: index * batch_size, :]
    else:
        x_train_batch = X_train[(index - 1) * batch_size: -1, :]
        y_train_batch = y_train[(index - 1) * batch_size: -1, :]
        index = 0
        repeat += 1

    index += 1

    sess.run(optimizer, feed_dict={X: x_train_batch, Y: y_train_batch})

    if i % mok == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={X: X_train, Y: y_train})
        print(repeat, ' :', 'loss : ', loss, 'acc : ', acc * 100, "+++")
        loss, acc = sess.run([cost, accuracy], feed_dict={X: X_test, Y: y_test})
        print(repeat, ' :', 'loss : ', loss, 'acc : ', acc * 100)

train_writer.close()
