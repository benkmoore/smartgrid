
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import statistics as stats


# fix random seed for reproducibility
np.random.seed(7)



# load data from previous (Julia) methods

train_q = pd.read_csv("global_q.csv", delimiter=",")
train_features = pd.read_csv("featureX.csv", delimiter=",")


## non visited
test_nv_q = pd.read_csv("global_q_nv.csv", delimiter=",")
test_nv_features = pd.read_csv("featureX_nv.csv", delimiter=",")

#all from Q
test_all_features = pd.read_csv("featureX_all.csv", delimiter=",")

Q_orig = pd.read_csv("Q_orig.csv", delimiter=",", header=None)


# split into input (X) and output (Y) variables
train_X = train_features
train_Y = train_q

####
####

#Scaled features


means = np.zeros((7,1))
stdev = np.zeros((7,1))
X_train_scaled = np.zeros((train_X.shape[0], train_X.shape[1]))
test_all_features_scaled = np.zeros((test_all_features.shape[0], test_all_features.shape[1]))

for i in range(0,7):
    means[i] = stats.mean(train_X.iloc[:,i])
    stdev[i] = stats.stdev(train_X.iloc[:,i])

    X_train_scaled[:,i] = train_X.iloc[:,i]-means[i]
    X_train_scaled[:,i] = X_train_scaled[:,i]*1/stdev[i]

    test_all_features_scaled[:,i] = test_all_features.iloc[:,i] - means[i]
    test_all_features_scaled[:,i] = test_all_features_scaled[:,i]*1/stdev[i]

### end loop

## Re-assign scaled values to test and train features




#get number of columns in training data
n_cols = train_X.shape[1]

nr_actions = Q_orig.shape[1]

model = Sequential()
model.add(Dense(7^2, activation='relu', input_shape=(n_cols,)))
# shape=(n_cols,) : this means n nr of input columns with any amount of rows, so open comma
#model.add(Dense(4, activation='sigmoid'))
model.add(Dense(7*2, activation='relu'))
model.add(Dense(7*3, activation='sigmoid'))
model.add(Dense(7*2, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Training the model using model fit
train_X = X_train_scaled
model.fit(train_X, train_Y, epochs=50)


# evaluate the model
scores = model.evaluate(train_X, train_Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


weights = model.get_weights() # list of numpy arrays
model.summary()

# Make predictions on dataset Xnew


test_q_predictions = model.predict(test_all_features_scaled)
## Save the predictions as .csv file
np.savetxt('test_q_predictions.csv', test_q_predictions, fmt='%.2f', delimiter=',')
np.savetxt('test_all_features.csv', test_all_features, fmt='%.2f', delimiter=',')
