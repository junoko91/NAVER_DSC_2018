import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

data = pd.read_csv("football_players_final.csv")

data = shuffle(data)

X = data.drop("Position",axis=1)
X = X.drop("Position_int",axis=1)
X = X.drop("Nationality",axis=1)
X = X.drop("Name",axis=1)
y = data["Position_int"]

X.head()
y.head()

# pca = PCA(n_components=15)
# X = pca.fit_transform(X)

# mdict = dict(enumerate(y.unique()))
# y.replace(to_replace=mdict.values(),value=mdict.keys(),inplace=True)
# Ability = X.values[:,5:-1]
# Ability = np.sum(Ability,axis=1)
# Ability = np.reshape(Ability,[len(X),1])
# Ability=StandardScaler().fit_transform(Ability)
X=StandardScaler().fit_transform(X)




# # row scaling
# X = X.T
# X = StandardScaler().fit_transform(X)
# X = X.T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# sm = SMOTE()
# X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

print(np.shape(X_train))
print(np.shape(y_train))
print(y_train.ravel())

y_test = np.reshape(np.asarray(y_test),[len(y_test),1])
y_train = np.reshape(np.asarray(y_train.ravel()),[len(y_train),1])

print(y.unique())

nbclass = len(y.unique())
nbfeature = np.shape(X_test)[1]

learning_rate = 0.0001
epoch = 200000
batch_size = 128
mok = int(len(X_train) / batch_size)

X = tf.placeholder(tf.float32,[None,nbfeature])
Y = tf.placeholder(tf.int32,[None,1])
Y_one_hot = tf.one_hot(Y,nbclass)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nbclass])

keep_prob = tf.placeholder(tf.float32)

size_l1 = 2048
size_l2 = size_l1/2
size_l3 =size_l2/2
size_l4 =size_l3/2

rate = 0.50
l1 = tf.layers.dense(X,size_l1,activation=tf.nn.relu,bias_initializer=tf.contrib.layers.variance_scaling_initializer())
l1 = tf.layers.dropout(l1,rate=rate)
l2 = tf.layers.dense(l1,size_l2,activation=tf.nn.relu,bias_initializer=tf.contrib.layers.variance_scaling_initializer())
l2 = tf.layers.dropout(l2,rate=rate)
l3 = tf.layers.dense(l2,size_l3,activation=tf.nn.relu,bias_initializer=tf.contrib.layers.variance_scaling_initializer())
l3 = tf.layers.dropout(l3,rate=rate)
l4 = tf.layers.dense(l3,size_l4,activation=tf.nn.relu,bias_initializer=tf.contrib.layers.variance_scaling_initializer())
l4 = tf.layers.dropout(l4,rate=rate)
l5 = tf.layers.dense(l1,nbclass,activation=None)

hypothesis = tf.nn.softmax(l5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l5,labels=Y_one_hot))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))

accuracy0 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(hypothesis, tf.argmax(Y_one_hot, 1), k=1), tf.float32))
accuracy2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(hypothesis, tf.argmax(Y_one_hot, 1), k=2), tf.float32))
accuracy3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(hypothesis, tf.argmax(Y_one_hot, 1), k=3), tf.float32))

is_top1 = tf.equal(tf.nn.top_k(hypothesis, k=1)[1][:, 0], tf.cast(tf.argmax(Y_one_hot, 1), tf.int32))
is_top2 = tf.equal(tf.nn.top_k(hypothesis, k=2)[1][:, 1], tf.cast(tf.argmax(Y_one_hot, 1), tf.int32))
is_top3 = tf.equal(tf.nn.top_k(hypothesis, k=3)[1][:, 2], tf.cast(tf.argmax(Y_one_hot, 1), tf.int32))
is_top4 = tf.equal(tf.nn.top_k(hypothesis, k=4)[1][:, 2], tf.cast(tf.argmax(Y_one_hot, 1), tf.int32))
is_in_top1 = is_top1
is_in_top2 = tf.logical_or(is_in_top1, is_top2)
is_in_top3 = tf.logical_or(is_in_top2, is_top3)
is_in_top4 = tf.logical_or(is_in_top3, is_top4)

accuracy11 = tf.reduce_mean(tf.cast(is_in_top1, tf.float32))
accuracy22 = tf.reduce_mean(tf.cast(is_in_top2, tf.float32))
accuracy33 = tf.reduce_mean(tf.cast(is_in_top3, tf.float32))
accuracy44 = tf.reduce_mean(tf.cast(is_in_top4, tf.float32))

print("train start")

position_dict = {0:"GK",1:"LB",2:"CB",3:"RB",4:"LM",5:"CM",6:"RM",7:"LW",8:"RW",9:"FW"}

saver = tf.train.Saver()

def train_start():
    index = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch * mok):
            if index <= mok:
                x_train_batch = X_train[(index - 1) * batch_size: index * batch_size, :]
                y_train_batch = y_train[(index - 1) * batch_size: index * batch_size, :]
            else:
                x_train_batch = X_train[(index - 1) * batch_size: -1, :]
                y_train_batch = y_train[(index - 1) * batch_size: -1, :]
                index = 0

            index += 1

            sess.run(optimizer, feed_dict={X: x_train_batch, Y: y_train_batch})

            if i % mok == 0:
                loss, acc = sess.run([cost, accuracy11], feed_dict={X: X_train, Y: y_train})
                print("<", i, ' :', 'loss : ', loss, ' acc : ', acc * 100, ">")
                if int(acc*100) >= 90:
                    loss, acc,hypo = sess.run([cost, accuracy33,hypothesis], feed_dict={X: X_test, Y: y_test})
                    print(i, ' :', 'loss : ', loss, ' acc : ', acc * 100)
                    top3 = tf.nn.top_k(hypo, k=3)[1].eval()
                    for ii in range(3):
                        print(ii," : ",position_dict[top3[0][ii]]," ")
                    hypo = hypo*1000
                    hypo = np.sort(hypo,axis=1)
                    saver.save(sess,"./model/kor.model.ckpt")
                    break
                else:
                    loss, acc = sess.run([cost, accuracy33], feed_dict={X: X_test, Y: y_test})
                    print(i, ' :', 'loss : ', loss, ' acc : ', acc * 100)


def model_set_before_start(model_name:str):
    model_set_before_start.last_model_name = ""
    model_set_before_start.sess = None

    if model_set_before_start.sess is not None and model_name == model_set_before_start.last_model_name:
        return model_set_before_start.sess
    elif model_set_before_start.sess is None:
        model_set_before_start.last_model_name = model_name
        model_set_before_start.sess = tf.Session()
        saver.restore(model_set_before_start.sess, model_name)
        model_set_before_start.sess.run(tf.global_variables_initializer())
    return model_set_before_start.sess

def predict_and_add(x:np.ndarray,model_name:str,list_dict):
    sess = model_set_before_start(model_name)
    feed_x = x[:,1:]
    hypo = sess.run(l5, feed_dict={X: feed_x})
    hypo = hypo * 1000
    top3 = tf.nn.top_k(hypo, k=3)[1].eval(session=sess)
    for iii in range(len(x)):
        flags = [True,True,True,True]
        print(top3[iii])
        for ii in range(3):
            tmp_dict = {x[iii][0]:hypo[iii][top3[0][ii]]}
            if top3[iii][ii] == 0:
                if(flags[0]):
                    tmp_list = list_dict["gk"]
                    flags[0] = False
                    tmp_list.append(tmp_dict)
            elif top3[iii][ii] <=3:
                if(flags[1]):
                    tmp_list = list_dict["back"]
                    flags[1] = False
                    tmp_list.append(tmp_dict)
            elif top3[iii][ii] <=6:
                if(flags[2]):
                    tmp_list = list_dict["mid"]
                    flags[2] = False
                    tmp_list.append(tmp_dict)
            elif top3[iii][ii] <=9:
                if(flags[3]):
                    tmp_list = list_dict["fw"]
                    flags[3] = False
                    tmp_list.append(tmp_dict)

def comb(list_dict,back:int , mid:int , fw:int):
    comb_gk = list(combinations(list_dict["gk"],1))
    comb_back = list(combinations(list_dict["back"],back))
    comb_mid = list(combinations(list_dict["mid"],mid))
    comb_fw = list(combinations(list_dict["fw"],fw))

    tmp_dict_list = []
    for i in comb_gk:
        for j in comb_back:
            for k in comb_mid:
                for l in comb_fw:
                    tmp = {**i,**j,**k,**l}
                    # tmp = i+j+k+l
                    # 점수를 계산해서 tmp 맨마지막에 넣어줘야함
                    if len(tmp) == 11:
                        score = sum(tmp.values())
                        tmp["score"] = int(score)
                        tmp_dict_list.append(tmp)

    return tmp_dict_list

# train_start()

X1 = pd.read_csv("football_players_final_kor.csv")
X1 = X1.drop("Position",axis=1)
X1 = X1.drop("Position_int",axis=1)
X1 = X1.drop("Nationality",axis=1)
X1 = X1.drop("Name",axis=1)
list_dict = {"gk":list(),"back":list(),"mid":list(),"fw":list()}
predict_and_add(X1.values,"./model/kor.model.ckpt",list_dict)
tmp = comb(list_dict,4,4,2)
print(tmp)