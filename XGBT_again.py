#!/usr/bin/python
# encoding=utf8
import numpy as np
import xgboost as xgb
import tuning_xgboost
import pickle


def read(filename):
    data = []
    label = []
    for cnt, line in enumerate(open(filename)):
        line = line.strip().strip("\n").split("\t")
        if len(line) != 37:
            continue
        data.append(list(map(float, line[3:])))
        label.append(list(map(int, line[2])))
    return np.array(data), np.array(label)


def read_test(filename):
    data = []
    label = []
    positive = 0
    negative = 0
    for _, line in enumerate(open(filename)):
        line = line.strip().strip("\n").split("\t")
        if len(line) != 37:
            continue
        if line[2] == "1":
            data.append(list(map(float, line[3:])))
            label.append(list(map(int, line[2])))
            positive += 1
        if positive > 1000:
            data.append(list(map(float, line[3:])))
            label.append(list(map(int, line[2])))
            negative += 1
        if positive > 1000 and negative > 2000:
            break

    return np.array(data), np.array(label)


has_save_file = True
has_save_file = False
is_train = True
# is_train = False

print ('start running example to used customized objective function')
if has_save_file:
    print("--------has save file---------")
    dtrain = xgb.DMatrix('data/dtrain.buffer')
    dtest = xgb.DMatrix('data/dtest.buffer')

else:
    print("loading data")
    train_data, train_label = read('data/train.txt')
    test_data, test_label = read_test('data/test.txt')
    dtrain = xgb.DMatrix(train_data, train_label)
    dtest = xgb.DMatrix(test_data, test_label)
    dtrain.save_binary("data/dtrain.buffer")
    dtest.save_binary("data/dtest.buffer")
    print("loading end ")

# Booster
if is_train:
    # grid1 = {'n_estimators': [100, 200, 500], 'learning_rate': [0.05, 0.2, 0.3, 0.5]}
    # grid2 = {'max_depth': [2, 4, 5, 8], 'min_child_weight': [2, 4, 8, 15]}
    # grid3 = {'colsample_bylevel': [0.7, 0.9], 'subsample': [0.7, 0.9]}
    # grid4 = {'scale_pos_weight': [1, 5, 10]}
    # grid5 = {'reg_alpha': [1, 10]}
    # hyperlist_to_try = [grid1, grid2, grid3, grid4, grid5]
    #
    # booster = xgb.XGBClassifier()
    # print 'Run Simple Parameter Tuning\n________'
    # tuned_estimator, grid = tuning_xgboost.grid_search_tuning(train_data, train_label, testX=test_data,
    #                                                           testY=test_label,
    #                                                           hyperparameter_grids=hyperlist_to_try,
    #                                                           booster=booster)
    # print("best train score:", grid.best_score_)
    # print("----------------------------------")
    # tuned_parameters = tuned_estimator.get_params()
    # for parameter in tuned_parameters:
    #     print(parameter, tuned_parameters[parameter])
    #
    # print("------")
    # preds = grid.predict_proba(test_data)
    # pred = []
    # for i in range(len(preds)):
    #     pred.append(preds[i][1])
    # print("test score:", pred)
    # tp = sum(int((pred[i] >= 0.5) == test_label[i]) for i in range(len(pred)))
    # print("------")
    # print("test precision:", float(tp) / len(pred))
    #
    # with open('model/model_pickle_file', "w") as fp:
    #     print("---save pickle----")
    #     pickle.dump(grid, fp)

    print("---------------------train again------------------------")
    num_round = 50000
    parameter = {'booster': 'gbtree',
                 'n_estimators': 200,
                 'learning_rate': 0.5,
                 'max_depth': 8,
                 'min_child_weight': 2,
                 'subsample': 0.9,
                 'colsample_bylevel': 0.9,
                 'scale_pos_weight': 1,
                 'reg_alpha': 1,
                 'reg_lambda': 1,
                 'objective': 'binary:logistic'}
    bst = xgb.train(params=parameter, dtrain=dtrain, num_boost_round=num_round,
                    early_stopping_rounds=100, evals=[(dtest, 'eval')])
    pred = bst.predict(dtest)
    tp = sum(int((pred[i] >= 0.5) == test_label[i]) for i in range(len(pred)))
    print("------")
    print("test precision:", float(tp) / len(pred))

    bst.save_model('model/gbdt.model')
    print("-----save gbdt.model-----")


else:

    with open('model/model_pickle_file', "r") as fp:
        grid_load = pickle.load(fp)

    preds = grid_load.predict_proba(test_data)
    pred = []
    for i in range(len(preds)):
        pred.append(preds[i][1])
    print("test score:", pred)
    tp = sum(int((pred[i] >= 0.5) == test_label[i]) for i in range(len(pred)))
    print("------")
    print("test precision:", float(tp) / len(pred))
