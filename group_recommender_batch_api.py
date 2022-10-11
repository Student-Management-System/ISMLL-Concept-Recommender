from numpy import mean
import pandas as pd
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np
from itertools import zip_longest
from itertools import repeat, chain
import itertools
import json
from swagger_client.api_client import ApiClient
# For Multilabel k Nearest Neighbours
from skmultilearn.adapt import MLkNN

def read_from_file(json_file='testdata/recommender-export.json'):
    # Opening JSON file
    f = open(json_file)
    data = json.load(f)
    f.close()

    return read_from_json(data)

def read_from_json(data_as_json):
    """This simulates the webservices and turns the JSON file
       into an object.

        :param json_file: The path to the exported JSON file

        :return: deserialized object.
    """

    # Convert to data object
    api_client = ApiClient()
    return api_client._ApiClient__deserialize(data_as_json, 'RExportDto')

def collect_students(data):
    students = [(s.user_info.user_id, s.user_info.username) for s in data.students]
    return students

def collect_assignments(data):
    assignments = [(a.id, a.name, a.collaboration, a.points, a.type, a.collaboration) for a in data.assignments if a.state == 'EVALUATED']
    return assignments

def collect_current_groups(data):
    group_names = [g.name for g in data.groups]
    groups_dict = dict.fromkeys(group_names)

    for g in data.groups:
        groups_dict[g.name] = [m.username for m in g.members]

    return groups_dict

def collect_group_settings(data):
    return int(data.course.group_settings.size_min), int(data.course.group_settings.size_max)

def df_submission_results(data, students, assignments):
    columns = ['Assignment', 'Max Points'] + [s[1] for s in students]
    tmp_list = []

    for a in assignments:
        row = [a[1], a[3]]
        assignment_id = a[0]
        max_points = a[3]
        for s in data.students:
            achieved_percentage = [g.achieved_points_in_percent for g in s.grades if g.assignment_id == assignment_id]
            achieved_percentage = achieved_percentage[0] if len(achieved_percentage) == 1 else 0
            achieved_points = achieved_percentage * max_points / 100
            row.append(achieved_points)
        tmp_list.append(row)

    df = pd.DataFrame(tmp_list, columns=columns)
    return df

def df_group_per_assignment(data, students, assignments):
    columns = ['Assignment'] + [s[1] for s in students]
    tmp_list = []

    for a in assignments:
        row = ['Group in ' + a[1]]
        assignment_id = a[0]
        is_group_work = a[5] == 'GROUP'
        for s in data.students:
            if is_group_work:
                group = [g.group.name for g in s.grades if g.assignment_id == assignment_id]
                group = group[0] if len(group) == 1 else ""
                row.append(group)
            else:
                row.append('SINGLE')
        tmp_list.append(row)

    df = pd.DataFrame(tmp_list, columns=columns)
    return df

def df_current_group(data, students, groups_dict):
    columns = ['Assignment'] + [s[1] for s in students]
    tmp_list = ['Current Group']

    for s in students:
        for group_name, members in groups_dict.items():
            if s[1] in members:
                tmp_list.append(group_name)
                break;

    df = pd.DataFrame(data=[tmp_list], columns=columns)
    return df

#read the dataset
def read_data(data_as_json):
    #Objects
    data = read_from_json(json)
    students = collect_students(data)
    assignments = collect_assignments(data)
    groups_dict = collect_current_groups(data)
    min_group_size, max_group_size = collect_group_settings(data)

    # Partial dataframes
    df_results = df_submission_results(data, students, assignments)
    df_groups = df_group_per_assignment(data, students, assignments)
    df_current_groups = df_current_group(data, students, groups_dict)

    #processing dataframes
    df_groups_ = df_groups.transpose().reset_index()
    df_groups_ = df_groups_.rename(columns=df_groups_.iloc[0]).drop(df_groups_.index[0])
    df_groups_ = df_groups_.astype('category')
    df_groups_ = pd.get_dummies(df_groups_.iloc[:,1:])
    #pd.get_dummies(df_groups_, columns=[x for x in df_groups_.iloc[:,1:]])
    print(df_groups_)


    df_results = df_results.transpose().reset_index()
    df_results = df_results.rename(columns=df_results.iloc[0]).drop(df_results.index[0])
    df_results= df_results.astype({col: int for col in df_results.columns[1:]})

    df_current_groups_ = df_current_groups.transpose().reset_index()
    df_current_groups_ = df_current_groups_.rename(columns=df_current_groups_.iloc[0]).drop(df_current_groups_.index[0])


    #final dataframe
    result = pd.concat([df_groups_, df_results, df_current_groups_], axis=1, join="inner")
    #new
    result= result.loc[:, ~(result == result.iloc[0]).all()]
    result.columns = [c.replace(' ', '_') for c in result.columns]
    result = result.loc[:,~result.columns.duplicated()]

    #result = result.drop('Group_in_testat_03', 1)

    #result["Current_Group"] = result.apply(lambda x: x["Current_Group"]-1, axis=1)
    result['Current_Group'] = result['Current_Group'].astype('category')
    cat_columns = result.select_dtypes(['category']).columns
    result[cat_columns] = result[cat_columns].apply(lambda x: x.cat.codes)
    g = result.groupby('Current_Group')
    result.loc[g['Current_Group'].transform(lambda x: len(x) < 2).astype(bool), 'Current_Group'] = (np.nan)

    To_be_assigned = result[result['Current_Group'].isna()]
    Training_Dataset = result.dropna()
    df2_Tr = Training_Dataset.drop(['Current_Group', 'Assignment'], axis=1)
    df2_To = To_be_assigned.drop(['Current_Group', 'Assignment'], axis=1)
    df3_Tr = Training_Dataset[['Current_Group']].copy()
    df3_Tr = pd.get_dummies(data=df3_Tr, columns=['Current_Group'])
    X= df2_Tr.to_numpy()
    y= df3_Tr.to_numpy()
    Xtest= df2_To.to_numpy()
    return result, To_be_assigned, X, y, Xtest, min_group_size, max_group_size



############################################################
###                                                      ###
### Multi-Label Classification with Neural network model ###
###                                                      ###
############################################################
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(15, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(15, input_dim=n_inputs, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y, Xtest):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=2, n_repeats=6, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=100)
        # make a prediction on the test set
        y_hat = model.predict(X_test)
        # round probabilities to class labels
        y_hat = y_hat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, y_hat)
        # store result
        #print('>%.3f' % acc)
        results.append(acc)
    yhat = model.predict(Xtest)
        # round probabilities to class labels
    yhat = yhat.round()
    return yhat, results

def assign_to_groups(dataset, to_be_assigned, labels, Prefered_Group_size):
    count = dataset.groupby(['Current_Group'], sort=False).size().reset_index(name='Count')
    dic1= pd.Series(count.Count.values,index=count.Current_Group).to_dict()
    dic1 = {int(k):int(v) for k,v in dic1.items()}
    assigned_labels = np.around(labels, decimals=5, out=None)
    assigned_labels = assigned_labels.tolist()
    assigned_groups=list()
    for a in assigned_labels:
            try:
                c = next((i, el) for i, el in enumerate(a) if dic1[i] < Prefered_Group_size)
                dic1[c[0]] +=1
                assigned_groups.append(c[0])
            except:
                pass

    assigned_groups= [x+1 for x in assigned_groups]

    assigned = ['Group'+ '{}'.format(s) for s in assigned_groups]
    col_one_list = to_be_assigned['Assignment'].tolist()
    my_dict = dict(zip_longest(col_one_list, assigned))
    return my_dict

def create_new_groups(assigned_groups, Prefered_Group_size):
    keys = [k for k, v in assigned_groups.items() if v == None]
    res = [keys[i:i+Prefered_Group_size] for i in range(0, len(keys), Prefered_Group_size)]
    indices = [index for index, value in enumerate(res)]
    indices= [x+1 for x in indices]
    index = ['NewGroup'+ '{}'.format(s) for s in indices]
    m = list()
    for listElem in res:
        count = len(listElem)
        m.append(count)
    final_indices = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(index, m))))
    new_dict = dict(zip_longest(keys, final_indices))
    return new_dict

def group_recommendation_with_neuronal_network(data_as_json, Prefered_Group_size=3):
    # load training and testing data
    df, Last, X, y, Xtest, min_group_size, max_group_size = read_data(data_as_json)
    X = np.asarray(X).astype('float32')
    Xtest = Xtest.astype(np.float)
    # evaluate model
    yhat, results = evaluate_model(X, y, Xtest)
    #print(yhat)

    # summarize performance
    #print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))

    # load dataset
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    model.fit(X, y, verbose=0, epochs=100)
    #preferred_group_size = int(input("Enter size: "))
    #Assign students to groups
    my_dict = assign_to_groups(df, Last, yhat, max_group_size)
    new_dict = create_new_groups(my_dict, max_group_size)
    my_dict.update(new_dict)

    for key,value in my_dict.items():
        print("Student ID : {} , Assigned group : {}".format(key,value),  ', NewGROUP:', "New" in value)
    #print(my_dict)

    return my_dict

#######################################
###                                 ###
### Multilabel k Nearest Neighbours ###
###                                 ###
#######################################
def get_MLkNN_model(X, y, Xtest, k):
    """
         param: Data, Labels, Test data and the number of nearest neighbours
        
         return: Predicted Labels.
    """
    classifier = MLkNN(k)
    # train
    classifier.fit(X, y)
    # predict
    prop = classifier.predict_proba(Xtest)
    yhat = prop.toarray()
    return yhat

    
def group_recommendation_with_knearest_neighbours(data_as_json, Prefered_Group_size=3):
    # load training and testing data
    df, Last, X, y, Xtest, min_group_size, max_group_size = read_data(data_as_json)

    # evaluate model
    yhat = get_MLkNN_model(X, y, Xtest, 3)
    #print(yhat)

    my_dict = assign_to_groups(df, Last, yhat,Prefered_Group_size)
    new_dict = create_new_groups(my_dict, Prefered_Group_size)
    my_dict.update(new_dict)
    for key,value in my_dict.items():
        print("Student ID : {} , Assigned group : {}".format(key,value),  ', NewGROUP:', "New" in value)

    return my_dict
