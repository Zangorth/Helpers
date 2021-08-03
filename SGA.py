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
                
                model = LogisticRegression(max_iter=500)
            
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                fit_list.append(f1_score(y_test, predictions, average='micro'))
        
        return np.mean(fit_list)
        
    def parallel(self, workers=2):
        pool = Pool(workers)
        out = pool.map(self.fit, range(10))
        pool.close()
        pool.join()
        
        return out