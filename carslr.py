import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


sns.set(rc={"figure.figsize" : (11.7,8.27)})

#read data
data=pd.read_csv("cars.csv")

cpdata=data.copy(deep=True)
info=cpdata.info() 
desc=cpdata.describe()
pd.set_option('display.float_format',lambda x : "%.3f" %x)

pd.set_option('display.max_columns',6)

desc_o=cpdata.describe(include="O")

cols=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cpdata.drop(cols,axis=1,inplace=True)

dup_rows=cpdata.duplicated() # gives duplicate rows series
cpdata=cpdata.drop_duplicates(keep='first')

dup_rows=cpdata.duplicated() 

nullval=cpdata.isnull().sum()

vals=np.unique(cpdata["yearOfRegistration"])
yearsum=sum(vals[85:])
sns.regplot(data=cpdata,x="yearOfRegistration",y="price",fit_reg=False)
'''Range fixed 1950 upto 2019'''

valprice=cpdata["price"].value_counts()
sns.histplot(x=cpdata["price"],kde=False)
sns.boxplot(y="price",data=cpdata)
'''range 100 above and 150000 below'''

powerval=cpdata["powerPS"].value_counts()
sns.histplot(x=cpdata["powerPS"])
'''range 10 to 500'''

cpdata=cpdata[
        (cpdata.yearOfRegistration<=2019)
        &(cpdata.yearOfRegistration>=1950)
        &(cpdata.price>=100)
        &(cpdata.price<=150000)
        &(cpdata.powerPS<=500)
        &(cpdata.powerPS>=10)
        ]
cpdata["monthOfRegistration"]/=12
cpdata["monthOfRegistration"]=round(cpdata["monthOfRegistration"],2)
cpdata["age"]=2018-cpdata["yearOfRegistration"]+cpdata["monthOfRegistration"]
cpdata.drop(columns=["yearOfRegistration","monthOfRegistration"],inplace=True)

'''age made'''


sns.histplot(cpdata["age"])
sns.boxplot(cpdata["age"])

sns.boxplot(y=cpdata["price"])


sns.regplot(x="age",y="price",data=cpdata,scatter=True,fit_reg=False)
sellerdet=pd.crosstab(cpdata["seller"],"count")


#offerType
offer=cpdata["abtest"].value_counts()


abtestvals=cpdata["abtest"].value_counts()
sns.boxplot(x="abtest",y="price",data=cpdata)

cpdata.drop(columns=["seller","abtest","offerType"],inplace=True)
carsdata=cpdata.copy(deep=True)

corelation=carsdata.corr()

'''method 1'''
'''linear regression'''
'''drop all nan rows'''
cars=carsdata.dropna(axis=0)
#print(cars.info())

cars=pd.get_dummies(cars,drop_first=True)
x=cars.drop("price",axis=1)
y=cars["price"]
# prices=pd.DataFrame({"1.Before":y,"2. After":np.log(y)})
# prices.hist()
y=np.log(y)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size= 0.3,random_state=0)


base_pred=np.mean(test_y)
base_pred=np.repeat(base_pred,len(test_y))
def errorcalc(preddata,test):
    mse=mean_squared_error(test,preddata)
    mse=np.sqrt(mse)
    return mse
#rmse value
rmse=errorcalc(base_pred,test_y)

lgr=LinearRegression(fit_intercept=True)
model_lgr=lgr.fit(train_x,train_y)
pred_lgr=lgr.predict(test_x)
mse_lgr=errorcalc(pred_lgr,test_y)
# error dropped

test1_r=model_lgr.score(test_x,test_y)
train1_r=model_lgr.score(train_x,train_y)

#residuals
res=test_y-pred_lgr
sns.regplot(y=res,x=pred_lgr,fit_reg=False,data=carsdata)
