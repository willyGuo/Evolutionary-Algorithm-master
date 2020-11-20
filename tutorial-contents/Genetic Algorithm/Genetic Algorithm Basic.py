"""
Visualize Genetic Algorithm to find a maximum point in a function.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10           # DNA length DNA的長度
POP_SIZE = 10           # population size   人口大小
CROSS_RATE = 0.8         # mating probability (DNA crossover) 交配機率
MUTATION_RATE = 0.003    # mutation probability  突變機率
N_GENERATIONS = 5     #代數
X_BOUND = [0, 5]         # x upper and lower bounds


def F(x): return np.sin(10*x) + np.cos(2*x)*x    # to find the maximum of this function


# 計算適應度 
# find non-zero fitness for selection
def get_fitness(pred): 

    return pred + 1e-3 - np.min(pred)

# 將二進制DNA轉換為十進制並將其規格化為範圍（0，5）
# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): 
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]

# nature selection wrt pop's fitness
# 選擇合適的留下來
def select(pop, fitness):    
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())     
    return pop[idx]

# 交配
def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)     #隨機選取基因序號
        #隨機定義boolean
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]
        """
        把POP中第i_行中被選擇為true的元素替代掉parent的false元素
        將parent為True的元素，將被改變，False的元素不變.將被改變的元素改變成pop矩陣第 i_行裡面被選擇為true的元素，
        注意，parent的是cross_points,  pop 也是cross_points，必須保持一致，才可以實現交叉生成一個新的個體，這個個體是父母基因的交叉結果
        """
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
        
    return parent

#突變
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            #隨機交換元素0變1，1變0
            child[point] = 1 if child[point] == 0 else 0
    return child

# np.random.randint(low,high,size)  這邊是隨機生成100個長度為10的基因(基因序列只有0、1)
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

plt.ion()       # something about plotting  動態畫圖
x = np.linspace(*X_BOUND, 200)    #0到5之間生成200個點
plt.plot(x, F(x))     #把曲線圖畫上去

for _ in range(N_GENERATIONS):     #跑兩百代
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA
    # something about plotting
    if 'sca' in globals(): sca.remove()  
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)   #帶入fittness公式計算，看分數
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])  #找出分數最高的基因
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

plt.ioff(); plt.show()


