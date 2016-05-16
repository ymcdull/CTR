
# coding: utf-8

# In[191]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

np.set_printoptions(suppress=True)


# In[192]:

header  = ["Click", "Weekday", "Hour", "Timestamp", "Log Type", "User ID", "User‐Agent", "IP", "Region", "City", "Ad Exchange", "Domain", "URL", "Anonymous URL ID", "Ad slot ID", "Ad slot width", "Ad slot height", "Ad slot visibility", "Ad slot format", "Ad slot floor price (RMB/CPM)", "Creative ID", "Key Page URL", "Advertiser ID", "User Tags"]
print(header)


# In[193]:

trainfile = 'train.txt'
testfile = 'testhead.txt'

n = 2847802
s = 100000
skip = sorted(random.sample(xrange(n),n-s))

train = pd.read_csv(trainfile, header = None, sep = '\t', names = header, skiprows=skip)
#test = pd.read_csv(testfile, header = None, sep = '\t', names = header[1:])

### Show head lines of files
print(train.head())
#print(test.head())

### output to csv file
# train.head().to_csv('out.csv', sep = '\t')


# In[136]:

### Check the structure of input files
# print(train.describe())
# print(train.shape)


# In[7]:

#len(train[train["Click"] == 1])


# In[194]:

### Test for pd.get_dummies
### Expand Weekdays
weekdays = pd.get_dummies(train["Weekday"], prefix = "Weekday")
weekdays.head()


# In[195]:

### Expand Hour
hour = pd.get_dummies(train["Hour"], prefix = "Hour")
hour.head()


# In[196]:

### Expand OS and Browser
os_and_browser = train["User‐Agent"].str.split("_", expand = True)
os = os_and_browser[0]
browser = os_and_browser[1]
os = pd.get_dummies(os, prefix="OS")
browser = pd.get_dummies(browser, prefix="Browser")
browser.head()


# In[197]:

### Expand floor price
def price_transfer(x):
  if x == 0:
    return '0'
  elif x >= 1 and x <= 10:
    return '1-10'
  elif x >= 11 and x <= 50:
    return '11-50'
  elif x >= 51 and x <= 100:
    return '51-100'
  else:
    return 'larger'

### Directly use dataframe method 'apply' to process series, instead of using map and lambda
floor_price = train["Ad slot floor price (RMB/CPM)"].astype(int).apply(price_transfer)
# floor_price = map(lambda price: price_transfer(price), train["Ad slot floor price (RMB/CPM)"].astype(int))

floor_price = pd.get_dummies(floor_price, prefix = "Price")

floor_price.head()
# type(floor_price)




# In[68]:

# ### Expand User tags
# print train["User Tags"].head()
# user_list = []
# for tag in train["User Tags"]:
#     if tag == 'null':
#         pass
#     else:
#         taglist = tag.strip().split(',')
#         for t in taglist:
#             if t not in user_list:
#                 user_list.append(t)
        
# print user_list


# In[198]:

mytrain = pd.concat([weekdays, hour, os, browser, floor_price], axis = 1)
mytrain.shape
mytrain.head()


# In[211]:

### Logistic Regression
# mytrain = train.ix[:, [1,2,8,9,10,15,16,17,18,19]]
# mytest = test.ix[:, [0,1,7,8,9,14,15,16,17,18]]
#print(mytest.describe())

# newtrain = mytrain.astype(str)
# newtrain = pd.get_dummies(newtrain)
# newtrain.shape


X,x,Y,y = train_test_split(mytrain, train["Click"], test_size = 0.1)
     
lr = LogisticRegression(C=0.1)
lr.fit(X, Y)
# print("The result of validation test is : %f" % lr.score(x,y))
### Predict class label for test samples
result = lr.predict_proba(x)
# print result.tolist()

### Predict probability for each class label
#result = lr.predict_proba(mytest)

#print(result[:100, 1])


# In[212]:

print(result[:,1])


# In[222]:

x = result[:,1]
# print(y)

### AUC Evaluation
auc = roc_auc_score(y, x)
print("The result of AUC is: %f" % auc)


# In[155]:

### GBRT feature engineering
def city_transfer(x, dict):
    return dict[int(x)]


counts = train["City"].value_counts()
counts = dict(counts)
city = train["City"].apply(city_transfer, args=(counts,))

counts = train["Region"].value_counts()
counts = dict(counts)
region = train["Region"].apply(city_transfer, args=(counts,))
# print(city_transfer(1, counts))

mytrain = pd.concat([weekdays, city,region], axis = 1)
mytrain.head()


# In[160]:

### GBRT

'''
X,x,Y,y = train_test_split(mytrain, train["Click"], test_size = 0.1)
     
# lr = LogisticRegression(C=0.1)
# lr.fit(X, Y)
# print("The result of validation test is : %f" % lr.score(x,y))
### Predict class label for test samples
# result = lr.predict_proba(x)
# gbrt = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, Y)
gbrt = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05,max_depth=5).fit(X, Y)
result = gbrt.predict_proba(x)

### New added for gbrt auc evaluation
x = result[:,1]
auc = roc_auc_score(y, x)
print("The result of AUC for gbrt is: %f" % auc)
'''



# In[162]:


# In[121]:
