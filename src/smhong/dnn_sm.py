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
# data.info()

# Input / Output Data 분류
raw_Y = data["Position"]
raw_X = data.drop("Position", "index", axis=1)

# Class 개수 정의 및 OneHot encoding
nb_classes = len(raw_Y.unique())

# Predict Data One hot encoding
one_hot_targets = pd.get_dummies(raw_Y)
# print(one_hot_targets)

# Train/Test 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(raw_X.values, one_hot_targets.values, test_size=0.3, random_state=42)

# Feature 개수 정의
nbfeature = np.shape(X_test)[1]

# 나눠진 데이터 shape 확인 ( Training : (12059, 13) Test : (5169, 13) )
# print(Y_train.shape)
# print(Y_test.shape)
train_num = Y_train.shape[0]
test_num = Y_test.shape[0]

# X, Y placeholder 설정
X = tf.placeholder(tf.float32, [None, nbfeature])
Y = tf.placeholder(tf.int32, [None, nb_classes])

W = tf.Variable(tf.random_normal([nbfeature, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

hidden_layer = 256

# weights & bias for nn layers
W1 = tf.get_variable("W1", shape=[nbfeature, hidden_layer],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([hidden_layer]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[hidden_layer, hidden_layer],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([hidden_layer]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[hidden_layer, hidden_layer],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([hidden_layer]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[hidden_layer, hidden_layer],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([hidden_layer]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[hidden_layer, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L2, W5) + b5

# parameters 설정
learning_rate = 0.00001
training_epochs = 500000
batch_size = 5
total_batch = int(train_num / batch_size)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y_train))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_train, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train my model
for epoch in range(training_epochs):
    for i in range(1):
        feed_dict = {X: X_train, Y: Y_train, keep_prob: 0.7}
        sess.run(optimizer, feed_dict=feed_dict)
        c, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
        if epoch % 1000 == 0:
            print('Epoch:', '%06d' % (epoch + 1), 'cost =', '{:.9f}'.format(c), 'acc =', '{:.5f}'.format(acc * 100))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: X_test, Y: Y_test, keep_prob: 1}))


# Get one and predict
a = test_num
b = 0
for i in range(50):
   # print(X_test[i:i + 1])
    label = sess.run(tf.argmax(Y_test[i:i + 1], 1))
    print("Label: ", label)
    result = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: X_test[i:i + 1], keep_prob: 1})
    print("Predict : ", result)
    if label == result:
        b = b + 1

print("Test Acc : ", (b / a * 100))