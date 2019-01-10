# This system is an algorithm that recommends items by trying to
# find users that are similar to each other based on their item ratings

# Acquiring data
# wget -O ./data/moviedataset.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip
# unzip -o ./data/moviedataset.zip -d ./data

import tensorflow as tf
import numpy as np
import pandas as pd

# Loading data
movies_df = pd.read_csv('./data/ml-1m/movies.dat', sep='::', header=None)
ratings_df = pd.read_csv('./data/ml-1m/ratings.dat', sep='::', header=None)

movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# 3883 movies, while our ID's vary from 1 to 3952
movies_df['List Index'] = movies_df.index

# let's merge the ratings dataframe into the movies one
# drop the Timestamp, Title and Genres columns since we won't be needing it to make recommendations

# Merging movies_df with ratings_df by MovieID
merged_df = movies_df.merge(ratings_df, on='MovieID')
# Dropping unecessary columns
merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

# Group up by UserID
userGroup = merged_df.groupby('UserID')

# Amount of users used for training
amountOfUsedUsers = 1000
# Creating the training list
trX = []
# For each user in the group
for userID, curUser in userGroup:
    # Create a temp that stores every movie's rating
    temp = [0] * len(movies_df)
    # For each movie in curUser's movie list
    for num, movie in curUser.iterrows():
        # Divide the rating by 5 and store it
        temp[movie['List Index']] = movie['Rating'] / 5.0
    # Now add the list of ratings into the training list
    trX.append(temp)
    # Check to see if we finished adding in the amount of users for training
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1

hiddenUnits = 20
visibleUnits = len(movies_df)
vb = tf.placeholder("float", [visibleUnits])  # Number of unique movies
hb = tf.placeholder("float", [hiddenUnits])  # Number of features we're going to learn
W = tf.placeholder("float", [visibleUnits, hiddenUnits])

# Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))

# Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

# Learning rate
alpha = 1.0
# Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
# Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
# Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

err = v0 - v1
err_sum = tf.reduce_mean(err * err)

# Current weight
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
# Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)
# Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)
# Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
# Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)
# Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    epochs = 15
    batchsize = 100
    errors = []
    for i in range(epochs):
        for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
            batch = trX[start:end]
            cur_w = session.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_vb = session.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_nb = session.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            prv_w = cur_w
            prv_vb = cur_vb
            prv_hb = cur_nb
        errors.append(session.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_nb}))
        print (errors[-1])

    # Selecting the input user
    inputUser = [trX[75]]
    # Feeding in the user and reconstructing the input
    hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
    feed = session.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
    rec = session.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})

    scored_movies_df_75 = movies_df
    scored_movies_df_75["Recommendation Score"] = rec[0]
    scored_movies_df_75.sort_values(["Recommendation Score"], ascending=False).head(20)

    merged_df.iloc[75]
    movies_df_75 = merged_df[merged_df['UserID'] == 215]
    movies_df_75.head()

    # Merging movies_df with ratings_df by MovieID
    merged_df_75 = scored_movies_df_75.merge(movies_df_75, on='MovieID', how='outer')
    # Dropping unecessary columns
    merged_df_75 = merged_df_75.drop('List Index_y', axis=1).drop('UserID', axis=1)

    merged_df_75.sort_values(["Recommendation Score"], ascending=False).head(20)
    print(merged_df_75)
