#todo: scores based on variable inpupt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.lda import LDA

def make_groups_equal_size(X,y):
    X_true = X[y == 1]
    y_true = y[y == 1]
    X_false = X[y == 0]
    y_false = y[y ==0]
    
    np.random.shuffle(X_true)
    np.random.shuffle(X_false)
    l = min([len(y_true), len(y_false)])
    X_true[:l, :]
    X_false[:l, :]
    X_ret = np.concatenate([X_true[:l, :], X_false[:l, :]])
    y_ret = np.concatenate([y_true[:l], y_false[:l]])
    return X_ret, y_ret

assert(len(make_groups_equal_size(np.array([[1,2,3],[2,3,4],[3,4,5], [4,5,6]]), np.array([0,0,1,1]))[0]) == 4)
assert(len(make_groups_equal_size(np.array([[1,2,3],[2,3,4],[3,4,5], [4,5,6]]), np.array([0,0,0,1]))[0]) == 2)

def prediction_rates(X_in,y_in, classifier = None, n = 5, return_ = 'score', normalize_group_size = True, verbosity = 0):
    '''
    X: training data
    y: target values
    classifier: classifier to use
    '''
    if classifier is None:
        pass
    else:
        raise ValueError("Not supported")
    score_all = []
    score_0 = []
    score_1 = []
    classifier_ = []
    lv = 0
    
    X_train =  X_test =  y_train = y_test = []
    for x in range(n):
        #if bad data is selected (e.g. only one group): reselect
        while True:
            try:
                if normalize_group_size:
                    X, y = make_groups_equal_size(X_in,y_in)  
                else:
                    X, y = X_in, y_in                            
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=x)
                #classifier = LDA(n_components=2, shrinkage = 'auto', solver = 'eigen', )
                classifier = LDA(n_components=2, shrinkage = None, solver = 'svd', )
                
                classifier.fit(X_train, y_train)
                break
            except ValueError:
                if lv >= 10:
                    raise RuntimeError("Can't select data, that is accepted by estimator!")
                lv = lv + 1
                continue
            
        score_all.append(classifier.score(X_test,y_test))
        score_1.append(classifier.score(X_test[y_test == 1, :],y_test[y_test==1]))
        score_0.append(classifier.score(X_test[y_test == 0, :],y_test[y_test==0]))
        classifier_.append(classifier)

    score = (np.median(score_1) + np.median(score_0))
    
    if verbosity > 1:
        print('score all: max %f min %f mean %f' % (max(score_all), min(score_all), np.mean(score_all)))
        print('score 1:   max %f min %f mean %f' % (max(score_1), min(score_1), np.mean(score_1)))
        print('score 0:   max %f min %f mean %f' % (max(score_0), min(score_0), np.mean(score_0)))
    if verbosity > 0:    
        print('score: %f' % score)
        print ''
    
    if return_ == 'all': 
        return dict(score_all = score_all, score_0 = score_0, score_1 = score_1, classifier_ = classifier_)
    elif return_ == 'score':
        return(score)
    
lda_prediction_rates = prediction_rates