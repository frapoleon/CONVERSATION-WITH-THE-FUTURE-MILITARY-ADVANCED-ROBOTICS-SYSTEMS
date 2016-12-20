
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split


from sklearn.metrics import mean_squared_error


# In[2]:

data = pd.read_csv("indicators_new2.csv")


# In[6]:

data2 = pd.read_csv("indicators_new1.csv")
data2


# In[23]:

data2 = pd.read_csv("indicators_new1.csv")
inde_vari = data['Is.Technology.Existed'].values
de_vari = data2['Is.Turning.Point'].values
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
#accuracy_score(y_true, y_pred)
cv = cross_validation.KFold(len(inde_vari), n_folds=10, shuffle=True)
score = []
for traincv, testcv in cv:
    train = [[i] for i in de_vari[traincv]]
    test = [[i] for i in de_vari[testcv]]
    gnb = GaussianNB()
    gnb.fit(train, inde_vari[traincv])
    #classifier = nltk.NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    score.append(accuracy_score(gnb.predict(test),inde_vari[testcv]))
print 'accuracy:', np.mean(score)


# In[24]:

from sklearn import datasets, metrics
from sklearn.metrics import roc_curve, auc
#gnb = GaussianNB()
#train = [[i] for i in de_vari]

#gnb.fit(train, inde_vari)
pred = gnb.predict(test)
#print accuracy_score(pred,inde_vari[testcv])
false_positive_rate, true_positive_rate, thresholds = roc_curve(inde_vari[testcv], pred)
print false_positive_rate, true_positive_rate, thresholds
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate)


# In[14]:

metrics.roc_curve(inde_vari, pred, pos_label=2)


# In[29]:

traindata,testdata,trainlabel,testlabel = train_test_split(data.values[:,:-1],data.values[:,-1],test_size=0.33, random_state=11)


# In[30]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(traindata, trainlabel)


# In[31]:

error = 0
pre = gnb.predict(testdata)
for i in range(len(testdata)):
    if pre[i] != testlabel[i]:
        error = error + 1

print float(error)/len(testlabel)


# In[32]:

pre


# In[33]:

testlabel


# In[34]:

from sklearn.linear_model import LogisticRegression
logic = LogisticRegression()
logic.fit(traindata, trainlabel)


# In[35]:

error = 0
pre = logic.predict(testdata)
for i in range(len(testdata)):
    if pre[i] != testlabel[i]:
        error = error + 1

print float(error)/len(testlabel)


# In[36]:

pre


# In[37]:

testlabel


# In[38]:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(traindata, trainlabel)


# In[39]:

error = 0
pre = rf.predict(testdata)
for i in range(len(testdata)):
    if pre[i] != testlabel[i]:
        error = error + 1

print float(error)/len(testlabel)


# In[40]:

pre


# In[41]:

testlabel


# In[42]:

from sklearn.svm import SVC
svm = SVC(kernel="linear")
svm.fit(traindata, trainlabel)


# In[43]:

error = 0
pre = svm.predict(testdata)
for i in range(len(testdata)):
    if pre[i] != testlabel[i]:
        error = error + 1

print float(error)/len(testlabel)


# In[44]:

pre


# In[ ]:




# In[19]:

from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(traindata, trainlabel)


# In[20]:

error = 0
pre = clf.predict(testdata)
for i in range(len(testdata)):
    if ((pre>0.5)*1.0)[i] != testlabel[i]:
        error = error + 1

print float(error)/len(testlabel)


# In[21]:

((pre>0.5)*1.0)


# In[22]:

testlabel


# In[27]:

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[24]:

feature_names=np.array(data.columns.values.tolist()[:-1])


# In[25]:

sorted_idx


# In[26]:

feature_names


# In[ ]:



