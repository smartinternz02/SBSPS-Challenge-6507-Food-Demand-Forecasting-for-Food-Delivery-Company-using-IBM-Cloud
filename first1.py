import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV as ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()

#print(train.head())

test.head()

#print(test.head())

train.info()

#print(train.info())

train['num_orders'].describe()

#print(train['num_orders'].describe())

train.isnull().sum()

#print(train.isnull().sum())

#merging datasets with common columns

meal_info = pd.read_csv("meal_info.csv")
center_info = pd.read_csv("fulfilment_center_info.csv")

trainfinal = pd.merge(train,meal_info, on="meal_id", how="outer")
trainfinal = pd.merge(trainfinal, center_info, on="center_id", how="outer")
trainfinal.head()

#print(trainfinal.head())

trainfinal = trainfinal.drop(['center_id','meal_id'], axis=1)
trainfinal.head()

#print(trainfinal.head())

cols= trainfinal.columns.tolist()
#print(cols)

cols= cols[:2]+ cols[9:] + cols[7:9] + cols[2:7]
#print(cols)

trainfinal = trainfinal[cols]
trainfinal.dtypes
#print(trainfinal.dtypes)

lb1 = LabelEncoder()
trainfinal['center_type'] = lb1.fit_transform(trainfinal['center_type'])

lb2 = LabelEncoder()
trainfinal['category'] = lb1.fit_transform(trainfinal['category'])

lb3 = LabelEncoder()
trainfinal['cuisine'] = lb1.fit_transform(trainfinal['cuisine'])

trainfinal.head()

#print(trainfinal.head())

trainfinal.shape

#print(trainfinal.shape)

plt.style.use('fivethirtyeight')
plt.figure( figsize=(12,7))
sns.displot(trainfinal.num_orders,bins = 25)
plt.xlabel("num_orders")
plt.ylabel("Number of Buyers")
plt.title("num_orders Distribution")

plt.show()

#output 

trainfinal2 = trainfinal.drop(['id'], axis=1)
correlation = trainfinal2.corr(method='pearson')
columns = correlation.nlargest(8, 'num_orders').index
columns
print(trainfinal2)

correlation_map = np.corrcoef(trainfinal2[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()

features= columns.drop(['num_orders'])
trainfinal13 = trainfinal[features]
x = trainfinal13.values
y = trainfinal['num_orders'].values
trainfinal.head()

#print(trainfinal.head())

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.25)

#print(train_test_split)

XG = XGBRegressor()
XG.fit(x_train, y_train)
y_pred = XG.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics 
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))


LR = LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:' , 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))

EN = ElasticNet()
EN.fit(x_train, y_train)
y_pred = EN.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:' , 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))

DT= DecisionTreeRegressor()
DT.fit(x_train, y_train)
y_pred = DT.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))

KNN = KNeighborsRegressor()
KNN.fit(x_train,y_train)
y_pred = KNN.predict(x_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))

GB = GradientBoostingRegressor()
GB.fit(x_train, y_train)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))

import pickle
pickle.dump(DT,open('fdemand.pkl','wb'))

testfinal = pd.merge(test,meal_info, on="meal_id", how="outer")
testfinal = pd.merge(testfinal, center_info, on ="center_id", how="outer")
testfinal = testfinal.drop(['meal_id','center_id'], axis=1)

tcols = testfinal.columns.tolist()
tcols = tcols[:2] + tcols[8:] + tcols[6:8] +tcols[2:6]
testfinal = testfinal[tcols]

lb1 = LabelEncoder()
testfinal['center_type'] = lb1.fit_transform(testfinal['center_type'])

lb2 = LabelEncoder()
testfinal['category'] = lb1.fit_transform(testfinal['category'])

lb3 = LabelEncoder()
testfinal['cuisine'] = lb1.fit_transform(testfinal['cuisine'])

x_test = testfinal[features].values

pred = DT.predict(x_test)
pred[pred<0] = 0
submit = pd.DataFrame({
	'id' : testfinal['id'],
	'num_orders' : pred

})

#submit.to_csv("submission.csv", index=False)

#submit.describe()

#print(submit.describe())

from ibm_watson_machine_learning import APIClient
wml_credentials={"url":"https://us-south.ml.cloud.ibm.com","apikey":"HdW0sTUJV2dQyaLB_voIUYGOwWkXZAMSiEmoYzZEaqOu"}
client=APIClient(wml_credentials)

client

def space_name(client, space_name):
    space=client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']['name']==space_name)['metadata']['id'])


space_uid= space_name(client,'food_deploy')
#print(space_uid)

client.set.default_space(space_uid)

#client.software_specifications.list()

MODEL_NAME = 'FoodModel'
DEPLOYMENT_NAME = 'food_deploy'
Food_MODEL = DT

software_spec=client.software_specifications.get_uid_by_name("default_py3.8")



software_spec

# Set Python Version
software_spec_uid = client.software_specifications.get_id_by_name('default_py3.8')

# Setup model meta
model_props = {
    client.repository.ModelMetaNames.NAME: MODEL_NAME, 
    client.repository.ModelMetaNames.TYPE: 'scikit-learn_0.23', 
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid 
}


#Save model
model_details = client.repository.store_model(
    model=Food_MODEL, 
    meta_props=model_props, 
    training_data=x_train, 
    training_target=y_train
)

#print(model_details)

model_uid = client.repository.get_model_uid(model_details); model_uid

#client.connections.list_datasource_types()

# Set meta
deployment_props = {
    client.deployments.ConfigurationMetaNames.NAME:DEPLOYMENT_NAME, 
    client.deployments.ConfigurationMetaNames.ONLINE: {}
}

# Deploy
deployment = client.deployments.create(
    artifact_uid=model_uid, 
    meta_props=deployment_props 
)


deployment_uid = client.deployments.get_uid(deployment)
payload = {"input_data":
           [
               {"fields":x_test.tolist(), "values":x_test.tolist()}
           ]
          }
result = client.deployments.score(deployment_uid, payload); result



print(payload)