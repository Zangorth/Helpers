from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from multiprocessing import Pool
import seaborn as sea
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('error', category=ConvergenceWarning)

sea.set_style('whitegrid')

class GA:
    def __init__(self, x, y, pop, cv=[], model='logit', max_features=None):
        self.x, self.y = x, y
        self.pop, self.cv = pop, cv
        self.max_features = max_features
        self.model = model
        self.max = []
        

    ####################
    # Fitness Function #
    ####################
    def fit(self, i=0):
        max_features = self.max_features if self.max_features != None else self.x.shape[1]
        rows = self.x.loc[self.pop['rows'][i] == 1].index
        cols = self.pop['cols'][i]
        
        fit_list = []
        
        if sum(cols) > max_features:
            fit_list.append(0)
        
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
                
                if self.model == 'logit':
                    if 'solver' not in self.pop:
                        solver = 'lbfgs'
                    else:
                        solver = self.pop['solver'][i]
                    
                    if 'penalty' not in self.pop:
                        penalty = 'l2'
                    else:
                        penalty = self.pop['penalty'][i]
                    
                    model = LogisticRegression(penalty=penalty, solver=solver)
                    
                    try:
                        model.fit(x_train, y_train)
                        predictions = model.predict(x_test)
                        fit_list.append(f1_score(y_test, predictions, average='micro'))
                    except ConvergenceWarning:
                        fit_list.append(0)
        
        return np.mean(fit_list)
    
    ###################
    # Mating Function #
    ###################
    def mate(self, fitness, num_parents):
        return self.pop.loc[self.pop.index.isin(((-np.array(fitness)).argsort())[0:num_parents])]
    
    #####################
    # Children Function #
    #####################
    def children(self, fitness, num_children, mutation_rate):
        children = pd.DataFrame(columns=self.pop.columns, index=range(num_children))
        
        for i in range(num_children):
            mates = np.random.choice(np.arange(0, len(self.pop)), size=2, p=fitness/sum(fitness))
            mom, dad = self.pop.iloc[mates[0]], self.pop.iloc[mates[1]]
            
            for j in range(len(self.pop.columns)):
                col = self.pop.columns[j]
                if type(mom[col]) == int:
                    mutate = bool(np.random.binomial(1, mutation_rate))
                    children.iloc[i, j] = round(np.mean([mom[col], dad[col]])) if mutate else np.random.choice([mom[col], dad[col]])
                    
                elif type(mom[col]) == float:
                    mutate = bool(np.random.binomial(1, mutation_rate))
                    children.iloc[i, j] = np.mean([mom[col], dad[col]]) if mutate else np.random.choice([mom[col], dad[col]])
                    
                elif type(mom[col]) == str:
                    mutate = bool(np.random.binomial(1, mutation_rate))
                    children.iloc[i, j] = str(np.random.choice(list(set(self.pop[col]))) if mutate else np.random.choice([mom[col], dad[col]]))
                
                elif type(mom[col]) == np.ndarray:
                    mutate = np.random.binomial(1, mutation_rate, len(mom[col]))
                    choice = np.random.binomial(1, 0.5, len(mom[col]))
                    child = [mom[col][i] if choice[i] == 1 else dad[col][i] for i in range(len(choice))]
                    children.iloc[i, j] = np.array([1 - child[i] if mutate[i] == 1 else child[i] for i in range(len(mutate))])
                
                else:
                    print('Invalid Data Type')
                    print(f'{mom[col]} needs to be data type int, float, str, or ndarray')
                    print(f'{mom[col]} is currently data type {type(mom[col])}')
                    print('')
                    break
                    
        return children
        
    #######################
    # Parallelize Fitness #
    #######################
    def parallel(self, solutions, workers=2):
        pool = Pool(workers)
        out = pool.map(self.fit, range(solutions))
        pool.close()
        pool.join()
        
        return out
    
    ##################
    # Model Function #
    ##################
        
    
    