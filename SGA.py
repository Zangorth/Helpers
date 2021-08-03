from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from multiprocessing import Pool
import numpy as np

class GA:
    def __init__(self, x, y, pop, cv=[], max_features=None):
        self.x, self.y = x, y
        self.pop, self.cv = pop, cv
        self.max_features = max_features
        
        
    def fit(self, i):
        max_features = self.max_features if self.max_features != None else self.x.shape[1]
        rows = self.x.loc[self.pop['rows'] == 1].index
        cols = self.pop['cols']
        
        fit_list = []
        
        if sum(cols) > max_features:
            fit_list.append(-666)
        
        else:
            x = self.x.dropna()
            y = self.y.loc[x.index]
            
            x = x[[x.columns[i] for i in range(x.shape[1]) if cols[i] == 1]]
            
            
            for fold in cv:
                x_test = x.loc[x.index.isin(fold)]
                
                x_train = x.loc[~x.index.isin(fold)]
                x_train = x_train.loc[~x_train.index.isin(rows)]
                
                y_test = y.loc[x_test.index]
                y_train = y.loc[x_train.index]
                
                model = LogisticRegression(max_iter=500)
            
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                fit_list.append(f1_score(y_test, predictions, average='micro'))
        
        return np.mean(fit_list)
        
    def parallel(self):
        pool = Pool()
        out = pool.map(self.fit, range(10))
        pool.close()
        pool.join()
        
        return out


ga = GA(x, y, pop, cv)

if __name__ == '__main__':
    print(ga.parallel())
    
    

pop = {'rows': np.random.binomial(1, 0.95, len(x)), 'cols': np.random.binomial(1, 0.95, x.shape[1])}
cv = [x.loc[semi == 'semi'].groupby(y, group_keys=False).apply(lambda x: x.sample(min(len(x), 25))).sort_index().index for i in range(5)]