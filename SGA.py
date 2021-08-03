from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from multiprocessing import Pool
import pandas as pd
import numpy as np

class GA:
    def __init__(self, x, y, pop, cv=[], max_features=None):
        self.x, self.y = x, y
        self.pop, self.cv = pop, cv
        self.max_features = max_features
        
        
    def fit(self, i=0):
        max_features = self.max_features if self.max_features != None else self.x.shape[1]
        rows = self.x.loc[self.pop['rows'][i] == 1].index
        cols = self.pop['cols'][i]
        
        fit_list = []
        
        if sum(cols) > max_features:
            fit_list.append(-666)
        
        else:
            x = self.x.dropna()
            y = self.y.loc[x.index]
            
            x = x[[x.columns[j] for j in range(x.shape[1]) if cols[j] == 1]]      
            
            for fold in self.cv:
                x_test = x.loc[x.index.isin(fold)]
                
                x_train = x.loc[~x.index.isin(fold)]
                x_train = x_train.loc[~x_train.index.isin(rows)]
                
                y_test = y.loc[x_test.index]
                y_train = y.loc[x_train.index]
                
                if 'solver' not in self.pop:
                    solver = 'lbfgs'
                else:
                    solver = self.pop['solver'][i]
                
                if 'penalty' not in self.pop:
                    penalty = 'l2'
                else:
                    penalty = self.pop['penalty'][i]
                
                model = LogisticRegression(max_iter=500, penalty=penalty, solver=solver)
            
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                fit_list.append(f1_score(y_test, predictions, average='micro'))
        
        return np.mean(fit_list)
    
    def mate(pop, fitness, num_parents):
        return pop.loc[pop.index.isin(((-np.array(fitness)).argsort())[0:num_parents])]
    
    def children(pop, fitness, num_children, mutation_rate):
        children = pd.DataFrame(columns=pop.columns, index=range(num_children))
        for i in range(num_children):
            mates = np.random.choice(np.arange(0, len(pop)), size=2, p=fitness/sum(fitness))
            mom, dad = pop.iloc[mates[0]], pop.iloc[mates[1]]
            
            for j in range(len(pop.columns)):
                if type(mom[pop.columns[j]]) == int:
                    mutate = bool(np.random.binomial(1, mutation_rate))
                    children.iloc[i, j] = round(np.mean([mom, dad])) if mutate else np.random.choice(mom, dad)
                    
                elif type(mom[pop.columns[j]]) == float:
                    mutate = bool(np.random.binomial(1, mutation_rate))
                    children.iloc[i, j] = np.mean([mom, dad]) if mutate else np.random.choice(mom, dad)
                    
                elif type(mom[pop.columns[j]]) == str:
                    mutate = bool(np.random.binomial(1, mutation_rate))
                    children.iloc[i, j] = np.random.choice(set(pop[pop.columns[j]])) if mutate else np.random.choice(mom, dad)
                
                elif type(mom[pop.columns[j]]) == np.ndarray:
                    mutate = np.random.binomial(1, mutation_rate, len(mom[pop.columns[j]]))
                    choice = np.random.binomial(1, 0.5, len(mom[pop.columns[j]]))
                    child = [mom[pop.columns[j]][i] if choice[i] == 1 else dad[pop.columns[j]][i] for i in range(len(choice))]
                    children.iloc[i, j] = np.array([1 - child[i] if mutate[i] == 1 else child[i] for i in range(len(mutate))])
                
                else:
                    print('Invalid Data Type')
                    break
                    
        return children
        
        
    def parallel(self, solutions, workers=2):
        pool = Pool(workers)
        out = pool.map(self.fit, range(solutions))
        pool.close()
        pool.join()
        
        return out