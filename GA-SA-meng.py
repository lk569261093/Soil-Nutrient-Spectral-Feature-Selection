import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from deap import creator, base, tools, algorithms
from scipy.optimize import dual_annealing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import random

# 读取 Excel 数据
data = pd.read_excel('TN.xlsx')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义适应度函数
def fitness(individual, X, y):
    selected_features = np.where(np.array(individual) == 1)[0]
    if len(selected_features) == 0:
        return -np.inf,
    X_selected = X[:, selected_features]

    kf = KFold(n_splits=5)
    scores = []

    for train_index, test_index in kf.split(X_selected):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores),

# 遗传算法
def ga_feature_selection(X, y):
    n_features = X.shape[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.choice, a=[0, 1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness, X=X, y=y)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)

    return hof[0]

# 模拟退火算法
def sa_feature_selection(X, y):
    n_features = X.shape[1]
    bounds = [(0, 1) for _ in range(n_features)]

    def func(individual, sign=-1.0):
        binary_individual = [round(x) for x in individual]
        return sign * fitness(binary_individual, X, y)[0],

    result = dual_annealing(lambda x: func(x)[0], bounds, maxiter=100)
    binary_individual = [round(x) for x in result.x]

    return binary_individual

# 蒙特卡罗技巧排序光谱数据
def monte_carlo_ranking(X, y, n_iter=10000):
    n_features = X.shape[1]
    scores = np.zeros(n_features)

    for _ in range(n_iter):
        i = random.randint(0, n_features - 1)
        j = random.randint(0, n_features - 1)

        if i == j:
            continue

        X_i = X[:, i].reshape(-1, 1)
        X_j = X[:, j].reshape(-1, 1)
        
        kf = KFold(n_splits=5)

        score_i = 0
        score_j = 0

        for train_index, test_index in kf.split(X_i):
            X_train_i, X_test_i = X_i[train_index], X_i[test_index]
            X_train_j, X_test_j = X_j[train_index], X_j[test_index] 
            y_train, y_test = y[train_index], y[test_index]

            model_i = LinearRegression()
            model_i.fit(X_train_i, y_train)
            score_i += model_i.score(X_test_i, y_test)

            model_j = LinearRegression()
            model_j.fit(X_train_j, y_train)
            score_j += model_j.score(X_test_j, y_test)

        if score_i > score_j:
            scores[i] += 1
        elif score_j > score_i:
            scores[j] += 1

    top_10_indices = np.argsort(scores)[-10:][::-1]
    return top_10_indices

# 执行特征选择
ga_individual = ga_feature_selection(X_scaled, y)
sa_individual = sa_feature_selection(X_scaled, y)
mc_top_10_indices = monte_carlo_ranking(X_scaled, y)

print("GA selected features:")
print(np.where(np.asarray(ga_individual) == 1)[0])

print("SA selected features:")
print(np.where(np.asarray(sa_individual) == 1)[0])

print("Top 10 most important spectral features by Monte Carlo method:")
print(mc_top_10_indices)
           
