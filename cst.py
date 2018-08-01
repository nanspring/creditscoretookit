# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#读取文件，返回训练集和测试集
def readFile(fileName,integerColumn,deleteColumn):
    file=pd.read_excel(fileName)
    for item in list(file):
        if(item in deleteColumn): #不需要的列给删去
            file.drop(item,axis=1,inplace=True)
            continue
        if(item in integerColumn):
            file[item]=file[item].fillna(-1)#所有数字数据空值都为-1  
        else:
            file[item]=file[item].fillna('空')#所有文字数据空值为 无    
    Y=file['target']
    file.drop('target',axis=1,inplace=True)
    X=file
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    print(X_train)
    print(y_train)
    print('y train',sum(y_train)/len(y_train))
    return X_train,X_test,y_train,y_test



#计算证据权重和信息值
def calculate_woe_iv(X_train,y_train,k): 
    total_positive = sum(y_train) #总阳性数据 这里默认y值为0或1
    total_negative = len(y_train)-total_positive #总阴性数据
    base = np.log(total_negative/total_positive) #woe公式后半段 公式：ln(Ni/Pi)-ln(sumN/sumP)
    woe = []
    iv = []
    for item in list(X_train):
        information = 0
        X_item_array = np.array(X_train[item]) #将dataframe转成array，方便走循环
        y_array = np.array(y_train) #y值同理
        property_num = len(set(X_train[item]))#set为计算属性单独值
        if(property_num <15): #以15为分界线，如果一个属性有大于15个的值，则被认为是连续，需要进行分箱，否则视为离散 分界值可根据数据需求进行调整
            item_property = set(X_train[item]) #算这个属性的离散值有多少个
            judge = [] #存储 【属性值，woe值 】
            for value in item_property: #循环走完属性的所有值 比如说性别 男，女 那么value == 男，女
                good,bad = 0,0 #初始阴性，阳性数据为0
                for i in range(len(X_item_array)): #把这个属性的训练数据走一遍，如果等于这个属性值，利用y值，求得阴性和阳性数据的量
                    if(X_item_array[i] == value):
                        if(y_array[i] == 0):
                            good = good + 1
                        else:
                            bad = bad + 1
                    if(good == 0): #如果有阴阳值为0，则设为0.0001
                        good = 0.0001
                    if(bad == 0):
                        bad = 0.0001
                current_woe = np.log(good/bad)-base
                current_iv = (good/total_negative-bad/total_positive)*current_woe
                information=information+current_iv
                judge.append([value, current_woe]) #ln(Ni/Pi)-ln(sumN/sumP)
            #print('judge ',judge)
            iv.append([item,information])
            woe.append([item,judge])
        else: #数据为连续性
            x=[]
            information = 0
            for i in X_item_array:#一维数据转成可用成kmean的形式
                x.append([i])
            y_predict = KMeans(n_clusters=k, random_state=42).fit_predict(x) #用kmean 分箱
            print(y_predict)
            count = np.zeros((k,2))
            judge = []
            for i in range(k):
                temp = []
                for j in range(len(y_predict)):
                    if(y_predict[j] == i):
                        temp.append(X_item_array[j])
                        if(y_array[j] == 0):
                            count[i][0] = count[i][0]+1 #好客户
                        if(y_array[j] == 1):
                            count[i][1] = count[i][1]+1 #坏客户
                if(count[i][0] == 0):
                    count[i][0] = 0.01
                if(count[i][1] == 0):
                    count[i][1] = 0.01
                current_woe = np.log(count[i][0]/count[i][1]) - base   
                current_information = (count[i][0]/total_negative - count[i][1]/total_positive)*current_woe
                information = information+current_information
                name = str(min(temp))+"~"+str(max(temp))
                judge.append([name,current_woe])
            woe.append([item,judge])
            iv.append([item,information])
    return woe,iv

                    

      

#stochastic gradient descent logistic regression
#logistic regression 


def accuracy(X_num,y_train,weight):
    c = 0
    for i in range(len(X_num)):
        x = X_num[i]
        x = np.append(x,1)
        y = y_train[i]
        result = sigmoid(np.dot(x,weight))
        if(result < 0.5):
            predict = 0
        elif(result > 0.5):
            predict = 1
        else:
            predict = -1
        if(predict == y):
            c = c+1
    accuracy = c/len(X_num)
    return accuracy

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def lossFunction(X_num,y_train,weight):
    N = len(X_num)
    loss = 0
    for i in range(N):
        x = X_num[i]
        x = np.append(x,1)
        x = np.array(x)
        h = sigmoid(np.dot(weight,x))
        loss = loss + y_train[i] * np.log(h) + (1-y_train[i])*np.log(1-h)
    return loss

def logisticRegression(X_num,y_train,maxIteration):
    X_num_copy = X_num.copy()
    weight = np.zeros(X_num_copy.shape[1]+1)
    alpha = 1
    count = 1
    N = len(X_num_copy)
    y_train = list(y_train)
    loss = []
    accuracy_list = []
    for j in range(maxIteration):
        alpha = alpha/count
        a = accuracy(X_num_copy,y_train,weight)
        accuracy_list.append(a)
        for i in range(N):
            index = random.randint(0,N-1)
            x = X_num_copy[index]
            y = y_train[index]
            x = np.append(x,1) #bias
            x = np.array(x)
            h = sigmoid(np.dot(x,weight))
            error = y - h
            weight = weight + alpha * error * x.transpose()
        count = count+1
        l = lossFunction(X_num_copy,y_train,weight)
        loss.append(l)
    plt.figure(1)
    plt.subplot(221)
    plt.plot(loss)
    plt.subplot(222)
    plt.plot(accuracy_list)
    plt.show()
    return weight

def logisticRegressionSk(X_num,y_train,X_test_num,y_test):
    lr = LogisticRegression(max_iter=50)
    lr.fit(X_num, y_train)
    print('SKlearn 逻辑回归参数 = ',lr.coef_)
    y_pred = lr.predict(X_test_num)
    print(y_pred)
    accuracy = (len(y_test)-sum(abs(y_test - y_pred)))/len(y_test)
    print('SKlearn精确度 = ',accuracy)
    return lr


#画KS图 和 ROC 图
def normalizeData(data):
    minS = (data[-1][0])
    maxS = (data[0][0])
    for i in range(len(data)):
        data[i][0] = (data[i][0]-minS)/(maxS-minS)*1000
        data[i][0] = 1000-data[i][0]
    return data

def ROC_KS_Graph(X_num,y_train,weight):
    X_num_KS = X_num.copy()
    y_train_KS = np.array(y_train.copy())
    KS_data = []
    total_bad = sum(y_train_KS)
    total_good = len(y_train_KS)-total_bad
    for i in range(len(X_num_KS)):
        x = np.array(X_num_KS[i])
        x = np.append(x,1)
        result = sigmoid(np.dot(x,weight))*1000
        KS_data.append([result,y_train_KS[i]])
    KS_data = sorted(KS_data,key = lambda KS_data:KS_data[0], reverse = True)
    temp = 0
    for i in KS_data:
        temp = temp + i[1]
    size = 80 #每个组有80条数据
    scale = int(len(KS_data)/size)
    KS_list = []
    for i in range(scale):
        temp_list = []
        for j in range(i*size,(i+1)*size):
            temp_list.append(KS_data[j][1])
        KS_list.append(temp_list)
    if(len(KS_data)%size !=0): #如果有多余的数据，全部加到最后一个组里面
        for i in range(size*scale,len(KS_data)):
            KS_list[len(KS_list)-1].append(KS_data[i][1])
    x_good = []
    y_bad = []
    g = 0
    b = 0
    for l in KS_list:
        bad = sum(l)
        good = len(l)-bad
        bad = bad/total_bad
        b = b+bad
        good = good/total_good
        g = g+good
        x_good.append(g*100)
        y_bad.append(b*100)
    plt.plot(x_good,y_bad)
    plt.plot(x_good,x_good,'r--')#红虚线做一个对角线的对比
    plt.title('ROC Curve(%)')
    plt.show()

    KS_data_copy = normalizeData(KS_data.copy())
    g = 0
    b = 0
    good = []
    bad = []
    score = []
    for i in range(20,1020,20):
        temp = KS_data_copy[:]
        for ks in temp:
            if(ks[0]>i):
                break
            else:
                if(ks[1] == 0):
                    KS_data_copy.remove(ks)
                    g = g+1
                elif(ks[1] == 1):
                    KS_data_copy.remove(ks)
                    b = b+1
                else:
                    print('数据错误 y值不仅只是0和1 还有其他值 ',ks[1])
        good.append(g/total_good)
        bad.append(b/total_bad)
        score.append(i)
    plt.subplot(111)
    plt.title('KS curve')
    plt.plot(score,good,'b--')
    plt.plot(score,bad,'r--')
    plt.legend(['good','bad'])
    plt.show()
    good = np.array(good)
    bad = np.array(bad)
    diff = bad - good
    KS_value = max(diff)*100
    print('KS 值 = ',KS_value)

def filter_by_iv(iv,X_train,X_test,threshold):
    iv = sorted(iv,key = lambda iv:iv[1], reverse = True)
    X_train_new = pd.DataFrame()
    X_test_new = pd.DataFrame()
    if(len(iv)<=threshold):
        return X_train,X_test
    else:
        for i in range(threshold):
            X_train_new[iv[i][0]] = X_train[iv[i][0]]
            X_test_new[iv[i][0]] = X_test[iv[i][0]]
        for i in range(threshold,len(iv)):
            print('根据信息值，已删除列 ',iv[i])
        return X_train_new,X_test_new


#将所有数据转化为数字 input：dataframe output：numpy array
def dataToNumber(X_train,woe): 
    X_data_df = X_train.copy() #dataframe
    X_data_temp = np.array(X_data_df) #将dataframe转换成array，这将去除列的名字
    x_dict = {} #用一个map记录列表的名字和对应的列序号
    c = 0
    for i in (list(X_data_df)): #添加列序号
        x_dict[i] = c
        c = c+1
    for i in range(len(woe)):
        try:
            woe_dict={}
            l=[] 
            X_temp = np.array(X_data_df[woe[i][0]]) #woe[i][0]是属性 比如性别 gender
            index = x_dict.get(woe[i][0]) #找到列名字对应的序号
            for j in range(len(woe[i][1])): #将属性所有的值对应的证据权重存到map里面
                woe_dict[woe[i][1][j][0]]= woe[i][1][j][1]
                if('~' in str(woe[i][1][j][0])): #检查看属性值是否为连续
                    l.append(woe[i][1][j][0].split('~'))
            if(len(l)>0):#如果l表不为空，则被认为是连续值 否则为离散
                for k in range(len(X_temp)):#循环整个训练集
                    for m in range(len(l)):
                        if(X_temp[k]>=float(l[m][0]) and X_temp[k]<=float(l[m][1])):
                            temp = l[m][0]+"~"+l[m][1]
                            X_data_temp[k][index] = woe_dict.get(temp,0)#将文本换成数字
            else:#离散值
                for k in range(len(X_temp)):#循环整个训练集
                    temp = X_temp[k]
                    X_data_temp[k][index] = woe_dict.get(temp,0)#将文本换成数字
        except:
            continue
    print(X_data_temp)
    return X_data_temp
   
#-------------------------------数据处理---------------------------------------------------------
integerColumn=['NumChildren','NumFamily','BoughtMoney','BoughtDate','WorkTime','MonthlyEarn','HETONGJINE','target']#为数值的列
deleteColumn=['GUID','WorkDate','DelayDay']#没用，需要删除的列
file_path = 'C:/Users/Administrator/Desktop/python/sample.xlsx' #文件路径
X_train,X_test,y_train,y_test=readFile(file_path,integerColumn,deleteColumn)#返回训练集和测试集
woe,iv = calculate_woe_iv(X_train,y_train,3) #计算证据权重和信息值，最后一个参数为连续数据的分箱数量
X_train,X_test= filter_by_iv(iv,X_train,X_test,15)#最后一个数值c代表至少取多少个变量 取信息值前c的变量 如果本身数据变量小于要求 X不变
print('证据权重 ',woe)
print('信息值 ',iv)
X_num = dataToNumber(X_train,woe)  #将训练集转成数字
X_test_num = dataToNumber(X_test,woe)#将测试集转成数字
y_test = np.array(y_test)
y_train = np.array(y_train)
#---------------------------------逻辑回归------------------------------------------------------------
weight = logisticRegression(X_num,y_train,50)#逻辑回归计算参数，最后一个参数为迭代量 这个时间会比较久 自己写的比较慢
print('逻辑回归参数 ',weight)
print('测试集的精确度 ',accuracy(X_test_num,y_test,weight))#看看测试集的精准度为多少
#！！！！！如果不用自己写的，用第三方包裹，就不要run上面三行的代码，直接跑下面四行的
lr = logisticRegressionSk(X_num,y_train,X_test_num,y_test)#用第三方包裹算得逻辑回归 这个会很快
weight = lr.coef_
weight = np.append(weight,lr.intercept_)
ROC_KS_Graph(X_num,y_train,weight)#画roc和ks图 计算ks值
#-------------------------------随机森林------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=25)
model = clf.fit(X_num,y_train)
print('随机森林参数：',model.feature_importances_)
predict = clf.predict(X_test_num)
print(predict)
rf_accuracy =( len(predict)-sum(abs(predict - y_test)) )/len(predict)
print('random forest accuracy = ',rf_accuracy)

#-----------------------------GBDT-------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators=6,learning_rate=0.2)
gbdt.fit(X_num, y_train)
guesslabels = gbdt.predict(X_test_num)
gbdt_accuracy =( len(guesslabels)-sum(abs(guesslabels - y_test)) )/len(guesslabels)
print('GBDT accuracy = ',gbdt_accuracy)


