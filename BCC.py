import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")
import time
from scipy.stats import pearsonr

def convert_to_one_hot(y):
    one_hot = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        index = int(y[i]-1)
        one_hot[i][index] = 1
    return one_hot

def loss(A,predicts,y_train_one_hot):

    fun = np.linalg.norm((np.sum((predicts * A.reshape(1,-1,1)), axis = 1) - y_train_one_hot))
    a = np.sum((predicts * A.reshape(1,-1,1)), axis = 1)
    
    return fun#np.linalg.norm((predicts * A.reshape(1,-1,1)) - y_train_one_hot)

def DSCombination(Dic1, Dic2):
    ## extract the frame dicernment      
    sets=set(Dic1.keys()).union(set(Dic2.keys()))
    Result=dict.fromkeys(sets,0)
    ## Combination process
    for i in Dic1.keys():
        for j in Dic2.keys():       
            if set(str(i)).intersection(set(str(j))) == set(str(i)):
                Result[i]+=Dic1[i]*Dic2[j]
            elif set(str(i)).intersection(set(str(j))) == set(str(j)):
                Result[j]+=Dic1[i]*Dic2[j]
                 
     ## normalize the results
    f= sum(list(Result.values()))
    for i in Result.keys():
        Result[i] /=f
    return Result
    
class Our:
    def __init__(self, data, n):
        super(Our, self).__init__()

        self.data = np.loadtxt(data)
        self.n = n

        for i in range(self.data.shape[1]-1):
            name = 'classifier'+str(i)
            setattr(self, name, KNeighborsClassifier())
    
    def main(self):
        seed_data = self.data
        np.random.shuffle(seed_data)

        h, w = seed_data.shape
        x_train = seed_data[:h//2,:-1]#划分训练集
        y_train = seed_data[:h//2,-1]
        x_test = seed_data[h//2:,:-1]#划分测试集
        y_test = seed_data[h//2:,-1]

        #1.训练集与测试集随机缺失数据
        for i in range(x_train.shape[0]):
            index=random.sample(range(x_train.shape[1]),self.n)
            x_train[i][index] = np.nan

        for i in range(x_train.shape[0]):
            index=random.sample(range(x_train.shape[1]),self.n)
            x_test[i][index] = np.nan
        
        #2.用训练集每一列的完整数据训练k个分类器
        A = np.random.rand(w-1)
        scores = np.random.rand(w-1)

        for i in range(x_train.shape[1]):
            index = np.argwhere(np.isnan(x_train[:,i]))
            index = list(set(np.arange(h//2))-set(index.reshape(-1)))
            x_vertical = x_train[:,i][index].reshape(-1,1)
            y_vertical = y_train[index].reshape(-1,1)

            score = pearsonr(x_vertical.reshape(-1), y_vertical.reshape(-1))[0]
            scores[i] = abs(score)

            name = 'classifier'+str(i)
            c = getattr(self,name)
            c.fit(x_vertical, y_vertical)
            acc = c.score(x_vertical, y_vertical)
            A[i] = acc
        A = A/np.sum(A)
        scores = (scores/np.sum(scores)).reshape(-1,1)

        #A = A*scores

        #3.在训练集上优化矩阵X
        #在训练集上对每个数据生成预测结果,同时按照index生成one_hot标签
        y_train_one_hot = convert_to_one_hot(y_train)
        #y_train_one_hot = y_train_one_hot.reshape(y_train_one_hot.shape[0],1,-1)
        #print(y_train_one_hot)
        #y_train_one_hot = np.repeat(y_train_one_hot, 7, axis=1)

        predicts = np.zeros((x_train.shape[0],x_train.shape[1],2))
        for i in range(x_train.shape[0]):
            index = np.argwhere(np.isnan(x_train[i]))                  #缺失列

            #y_train_one_hot[i][index] = 0                              #将缺失列置0，在计算cost时候不计入计算

            index = list(set(np.arange(x_train.shape[1]))-set(index.reshape(-1)))     #完整列

            scores_index = scores[index]
            scores_index = scores_index/np.sum(scores_index)

            for j in index:
                name = 'classifier'+str(j)
                c = getattr(self,name)
                predicts[i,j] = c.predict_proba(x_train[i][j].reshape(-1,1))
            predicts[i] = scores * predicts[i]
            predicts[i] = predicts[i] / (np.sum(predicts[i],axis = 1)).reshape(-1,1)
            predicts[i][np.isnan(predicts[i])]=0.
        
        #随机初始化矩阵A

        #使用BFGS优化
        e = 1e-2
        cons = ({'type': 'ineq', 'fun': lambda x: x - e},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        res = minimize(loss, A, args = (predicts,y_train_one_hot), method = 'L-BFGS-B' ,constraints = cons)
        #opt.minimize(目标函数, 初始值, constraints=约束条件, bounds=约束边界, jac=雅可比函数, ...)   #去掉了method = 'BFGS',
        res_A = res.x

        #4.测试
        #获取测试结果
        predicts_test = np.zeros((x_test.shape[0],x_test.shape[1],2))
        index_test = []
        for i in range(x_test.shape[0]):
            index = np.argwhere(np.isnan(x_test[i]))
            index = list(set(np.arange(x_train.shape[1]))-set(index.reshape(-1)))
            index_test.append(index)
            scores_index = scores[index]
            scores_index = scores_index/np.sum(scores_index)
            for j in index:
                name = 'classifier'+str(j)
                c = getattr(self,name)
                predicts_test[i,j] = c.predict_proba(x_test[i][j].reshape(-1,1))
            predicts_test[i] = scores * predicts_test[i]
            predicts_test[i] = predicts_test[i] / (np.sum(predicts_test[i],axis = 1)).reshape(-1,1)
            predicts_test[i][np.isnan(predicts_test[i])]=0.
        
        #对每一个测试集的测试结果与权重融合
        pre = []
        for i in range(len(index_test)):
            A_i = res_A[index_test[i]].reshape(1,-1)
            A_i = A_i/np.sum(A_i)
            
            predicts_test_i = predicts_test[i][index_test[i]]
            mass = A_i.reshape(-1,1) * predicts_test_i
            mass_all = 1 - A_i
            new_array = np.hstack((mass, np.transpose(mass_all)))

            key_alone = [chr(i) for i in range(97,(97+2))]
            key_all = [''.join([chr(i) for i in range(97,(97+2))])]
            key = key_alone+key_all
            dict_0 = dict(zip(key,new_array[0]))
            if new_array.shape[0] == 1:
                dict_0.pop(''.join(key_all))
                label_result = max(dict_0,key=lambda x:dict_0[x])
                pre.append(ord(label_result)-97)
            else:
                num = 0
                for q in new_array[1:]:
                    num+=1
                    dict_ = dict(zip(key,q))
                    if num == 1:
                        result = DSCombination(dict_0,dict_)
                    result = DSCombination(result,dict_)
                result.pop(''.join(key_all))
                label_result = max(result,key=lambda x:result[x])
                pre.append(ord(label_result)-97)

        total = 0
        for i in range(y_test.shape[0]):
            if (y_test[i]-1) == pre[i]:
                total+=1
        acc = total/(y_test.shape[0])

        return acc

if __name__ == "__main__":
    ta = time.time()
    data = 'cleve.txt'
    index = [4,6,8]
    for n in index:
        accs = []
        for i in range(5):
            our = Our(data, n)
            acc = our.main()
            accs.append(acc)
        print('缺失值n:',n,'平均值:',np.mean(accs),'+-标准差:',np.std(accs,ddof=1))
        print('\n')
    print(time.time()-ta)