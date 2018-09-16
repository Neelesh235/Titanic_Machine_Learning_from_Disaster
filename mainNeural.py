import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from  keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

onehot_encoder = OneHotEncoder(sparse=False)

label_encoder = LabelEncoder()

classifier = Sequential()

df = pd.read_csv('/home/neelesh/Documents/Machine Learning Competition/train.csv')

df = df.fillna("0")

X = df.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9]].values

Y = df.iloc[:, 11:12].values


X[:, 1] = label_encoder.fit_transform(X[:, 1])
X[:, 5] = label_encoder.fit_transform(X[:, 5])
X[:, 7] = label_encoder.fit_transform(X[:, 7])


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

st_sc = StandardScaler()
x_train = st_sc.fit_transform(x_train)
x_test = st_sc.fit_transform(x_test)

# Using logistic Regression to start with

classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=8))

classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

modelfit = classifier.fit(x_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_pred, y_test)

corr = 0
incorr = 0

for i in range(0, len(y_test)):
    if y_test[i] == y_pred[i]:
        corr += 1
    else:
        incorr += 1

corrPer = (corr / len(y_test)) * 100

incorPer = (incorr / len(y_test)) * 100


# use the model for test file

df_test = pd.read_csv('/home/neelesh/Documents/Machine Learning Competition/test.csv')

df_test = df_test.fillna("0")

x_maintest = df_test.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9]].values

y_maintest = df_test.iloc[:, 11:12].values


x_maintest[:, 1] = label_encoder.fit_transform(x_maintest[:, 1])
x_maintest[:, 5] = label_encoder.fit_transform(x_maintest[:, 5])
x_maintest[:, 7] = label_encoder.fit_transform(x_maintest[:, 7])

x_maintest = st_sc.fit_transform(x_maintest)

y_mainpred = classifier.predict(x_maintest)

y_mainpred = (y_mainpred > 0.5)

y_mainpred = [int(x == True) for x in y_mainpred]


corr = 0
incorr = 0

for i in range(0, len(y_maintest)):
    if y_maintest[i] == y_mainpred[i]:
        corr += 1
    else:
        incorr += 1

corrPer = (corr / len(y_maintest)) * 100

incorPer = (incorr / len(y_maintest)) * 100

df_created = pd.DataFrame(y_mainpred, columns=['Survived'])
df_created.to_csv("/home/neelesh/Documents/Machine Learning Competition/pred.csv")

pass



