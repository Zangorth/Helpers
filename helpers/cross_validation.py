from imblearn.over_sampling import SMOTE
from multiprocessing import Pool
from scipy.stats import sem
import numpy as np


class CV():
    def __init__(self, model, metric, splitter, lower_bound=False):
        self.model, self.metric = model, metric
        self.lower_bound = lower_bound
        self.splitter = splitter
        
        return None
    
    def cv_unitary(self, iterator):
        metric_list = []
        for i in range(self.cv):
            x_train, x_test, y_train, y_test = self.splitter(self.x, self.y)
            
            if self.over and self.workers == 1:
                oversample = SMOTE(n_jobs=-1)
                x_train, y_train = oversample.fit_resample(x_train, y_train)
            
            else:
                oversample = SMOTE()
                x_train, y_train = oversample.fit_resample(x_train, y_train)
    
            self.model.fit(x_train, y_train)
            predictions = self.model.predict(x_test)
            
            score = self.metric(y_test, predictions)
            metric_list.append(score)
            
        if self.full:
            return metric_list
        
        elif self.lower_bound:
            return np.mean(metric_list) - 1.96*sem(metric_list)
        
        else:
            return np.mean(metric_list)
        
    def cv_parallel(self):
        with Pool(self.workers) as pool:
            metric_list = pool.map(self.cv_unitary, range(self.solutions))
            pool.close()
            pool.join()
            
        if self.full:
            return metric_list
        
        elif self.lower_bound:
            return np.mean(metric_list) - 1.96*sem(metric_list)
        
        else:
            return np.mean(metric_list)
        
    
    def cv(self, x, y, cv=20, over=True, full=False, workers=1):
        self.x, self.y = x, y
        self.over, self.full = over, full
        self.workers=workers
        
        if workers == 1:
            self.cv = cv
            out = self.cv_unitary(iterator=0)
            
        elif workers > 1:
            self.solutions = cv
            self.cv = 1
            
            out = self.cv_parallel()
            
        else:
            return 'Number of Workers must be >= 1'
            
        del [self.x, self.y, self.cv, self.over, self.full, self.workers]
            
        return out