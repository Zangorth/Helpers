from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('error', category=ConvergenceWarning)

def fitness(x, y, time, max_features, skip_time, cut_dates, model, path, solution):
    metric_list = []
    
    if sum(solution['solution']) > max_features:
        metric_list.append(-666)
        
    else:
        if model == 'logit':
            solution = solution['solution']
            x = pd.read_feather(path, columns = [x[i] for i in range(len(x)) if solution[i] == 1])
            model = LogisticRegression(solver='saga', max_iter=300)
            
        elif model == 'boost':
            estimator = solution['estimator']
            learn = solution['learn']
            depth = solution['depth']
            solution = solution['solution']
            
            x = pd.read_feather(path, columns=[x[i] for i in range(len(x)) if solution[i] == 1])
            model = GradientBoostingClassifier(n_estimators=estimator, learning_rate=learn, max_depth=depth)
            
        for date in cut_dates:
            x_test, x_train = x.loc[(time > date) & (time <= date + relativedelta(months=skip_time))], x.loc[time <= date]
            y_test, y_train = y.loc[(time > date) & (time <= date + relativedelta(months=skip_time))], y.loc[time <= date]
            
            try:
                model.fit(x_train, y_train)
                predictions = model.predict_proba(x_test)[:, 1]
                metric_list.append(roc_auc_score(y_test, predictions))
            
            except ConvergenceWarning:
                metric_list = [-777]
                break
            
        return(np.mean(metric_list))
    
    
def mate(pop, fitness, num_parents):
    return pop[(-np.array(fitness)).argsort()][0:num_parents]

def crossover(parents, offspring_size, typ='variables', mutations=3, meth='unif', mn=0, mx=0):
    offspring = np.empty(offspring_size)
    
    for i in range(offspring_size[0]):
        choice = np.random.choice(np.arange(0, len(parents)), size=2)
        mom, dad = parents[choice[0]], parents[choice[1]]
        genes = np.random.binomial(1, p=0.5, size=len(mom))
        offspring[i] = np.where(genes == 1, mom, dad)
        
        prob = 0.001 if mutations/sum(offspring[i]) <= 0 else 0.999 if mutations/sum(offspring[i]) >= 1 else mutations/sum(offspring[i])
        switch_off = np.random.binomial(1, p=prob, size=len(offspring[i]))
        
        prob = 0.001 if mutations/(len(offspring[i]) - sum(offspring[i])) <= 0 else 0.999 if mutations/(len(offspring[i]) - sum(offspring[i])) else mutations/(len(offspring[i]) - sum(offspring[i]))
        switch_on = np.random.binomial(1, p=prob, size=len(offspring[i]))
        
        offspring[i] = np.where((offspring[i] == 1) & (switch_off == 1), 0, offspring[i])
        offspring[i] = np.where((offspring[i] == 1) & (switch_on == 1), 1, offspring[i])
        
    return offspring
        
    