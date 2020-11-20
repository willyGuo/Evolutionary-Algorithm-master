"""
Visualize Genetic Algorithm to find the shortest path for travel sales problem.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import random

N_CITIES = 280  # 城市個數
CROSS_RATE = 1   #交配機率
MUTATE_RATE = 1   #突變機率
POP_SIZE = 20 #人口數
N_GENERATIONS = 3    #代數
class GA2(object):
    def __init__(self, city_size):
        self.DNA_size = city_size
        #random.seed(1)
        self.pop_start = random.randint(1,280)   #隨機初始第一個開始的城市    
    
    def find2city(self, mp):
        threshold = 0.97
        remeber=[]
        
        for i, value in enumerate(mp[self.pop_start]):
            if value > threshold and value !=1:
                remeber.append(i)
        a = random.choice(remeber)
        return a
    def findnextcity(self, city, mp, travel):
        threshold = 0.95
        remeber=[]
        for i, value in enumerate(mp[city]):
            if value > threshold and value <1:
                remeber.append(i)
        while len(remeber) < 1:
            threshold - 0.5
            for i, value in enumerate(mp[city]):
                if value > threshold and value !=1:
                    remeber.append(i)

        a = random.choice(remeber)
        while a in travel:
            a =random.choice(remeber)
        return a
    
    #def select(self, parent):     
class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        #生成一個二維陣列，且數字為0~DNA_size之間，先隨機生成每個商人會走的路線
        #vstack為堆疊陣列
        #permutation為排列不重複
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):     
        line_x = np.empty_like(DNA, dtype=np.float64)   #生成一個跟DNA依樣大小的陣列
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]   #根據每位旅行者的路徑，重新排列城市座標的順序
            line_x[i, :] = city_coord[:, 0]   #將路徑中城市的X座標存成line_x的一維陣列
            line_y[i, :] = city_coord[:, 1]   #和上面相同，把y座標存為line_y的一維陣列
        return line_x, line_y   #回傳兩個陣列
    
    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)   #產生一個一維陣列
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):  #將x座標和y座標合起來成二維陣列
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))    #把xs和ys的(N+1) - N後平方，再相加，最後再開根號，此為取距離
        fitness = np.exp(self.DNA_size * 2 / total_distance)   #取高斯，讓距離差異度變大  
        return fitness, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]   #取代原族群

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:   #如果cross_rate比隨機數大
            i_ = np.random.randint(0, self.pop_size, size=1)       # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            keep_city = parent[~cross_points]             #只有true的會留下來，且往前放，false的就被刪掉了
            #np.isin(a,b) 用於判定a中的元素在b中是否出現過，如果出現過返回True,否則返回False,最終結果為一個形狀和a一模一樣的陣列。
            #ravel為轉一維陣列
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]   #把pop其中一個人口中和keep_city有重複的數據刪掉，留下沒有的
            parent[:] = np.concatenate((keep_city, swap_city))  #再把兩個陣列連接起來
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child   #代替掉原來的陣列
        self.pop = pop    #回傳更改後的所有旅行者走的路


class TravelSalesPerson(object):
    def __init__(self, n_cities):

        training_data = pd.read_excel("a280.xlsx", header=None)
        position_np = training_data.as_matrix()    #將陣列中決定城市各數，生成2維的座標，且正規化成0~1之間，且不重複(這邊在決定每個城市的座標)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.city_position = min_max_scaler.fit_transform(position_np)
        
    def calfitmp(self):    
        x_, y_  = zip(*self.city_position)
        x_ = np.array(x_)
        y_ = np.array(y_)

        B = np.vstack([ (x_[i]-x_) for i, value in enumerate(x_)])
        C = np.vstack([ (y_[i]-y_) for i, value in enumerate(y_)]) 
        B = np.square(B)
        C = np.square(C)
        D = B + C
        D = 1-(D / 1.6243)
        

        """
        data_df = pd.DataFrame(D) 
        writer = pd.ExcelWriter('Save_Excel.xlsx')
        data_df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
        writer.save()
        """
        #self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])
        #用for迴圈抓出每個點位，再計算抓出的點對其他點的差值
        plt.ion()   #打開動態畫圖
        return D
    def plotting(self, lx, ly, total_d,num):
        plt.cla()   #刪除上一次在圖上的資訊
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=5, c='k')   #畫出city_posistion中的位置
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 1, 'color': 'red'})
        plt.text(-0.05, 1, "Gen : %.2f" % num, fontdict={'size': 17, 'color': 'green'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)

#ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
travelroad = []    #生成一個空陣列
env = TravelSalesPerson(N_CITIES)   #加入城市座標，並壓縮距離成0~1之間，且將歸屬度表格計算出來
ga2 = GA2(city_size=N_CITIES )    #初始GA2
travelroad.append(ga2.pop_start)   #把第一個城市加到陣列
a = ga2.find2city(env.calfitmp())   #用歸屬度來找出下一個城市
travelroad.append(a)
print(travelroad)
print(travelroad[-1])
for i in range(2,N_CITIES):
    travelroad.append(ga2.findnextcity(city=travelroad[-1], mp=env.calfitmp(), travel=travelroad))
print("個數")
print(len(travelroad))
print(travelroad)
for i in range(0,280):
    print(travelroad.count(i))

"""
for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)   #回傳每個旅行者到的城市的X座標和Y座標
    fitness, total_distance = ga.get_fitness(lx, ly)      #計算適應度，和總長度
    ga.evolve(fitness)    #交配以及突變
    best_idx = np.argmax(fitness)     #找出陣列中最大值是哪個
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)  #SHOW迴圈中，最好的基因是哪一個
    print('Gen:', generation, '| best total_distance: %.2f' % total_distance[best_idx],)
    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx], generation)
    """
plt.ioff()
plt.show()
