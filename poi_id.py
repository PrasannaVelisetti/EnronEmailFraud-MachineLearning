import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# adding new features in the dataset
for k in data_dict.keys():
    data_dict[k]['from_messages_ratio']  = 0
    if (data_dict[k]['from_poi_to_this_person'] != 'NaN') and (data_dict[k]['from_messages'] != 'NaN') and (data_dict[k]['from_messages'] != 0):
        data_dict[k]['from_messages_ratio'] = float(data_dict[k]['from_this_person_to_poi'])/float(data_dict[k]['from_messages'])

    data_dict[k]['to_messages_ratio']  = 0
    if (data_dict[k]['from_this_person_to_poi'] != 'NaN') and (data_dict[k]['to_messages'] != 'NaN') and (data_dict[k]['to_messages'] != 0):
        data_dict[k]['to_messages_ratio'] = float(data_dict[k]['from_poi_to_this_person'])/float(data_dict[k]['to_messages'])
    
    data_dict[k]['TotalPayments-Salary']=0    
    if (data_dict[k]['salary'] != 'NaN') and (data_dict[k]['total_payments'] != 'NaN') and (data_dict[k]['salary'] != 0):
        data_dict[k]['TotalPayments-Salary']=float(data_dict[k]['total_payments'])/float(data_dict[k]['salary'])
        
    data_dict[k]['IsDirector']=0
    if (data_dict[k]['director_fees'] != 'NaN'):
        data_dict[k]['IsDirector']=1    
    
#This is the final list of features selected
#features_list = ['poi','salary','bonus','long_term_incentive','deferred_income','IsDirector','deferral_payments','other','expenses','IsDirector','total_payments','exercised_stock_options','restricted_stock','to_messages_ratio','from_messages_ratio']
features_list = ['poi','salary','bonus','long_term_incentive','deferred_income','expenses','IsDirector','total_payments','exercised_stock_options','to_messages_ratio','from_messages_ratio']


del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['FREVERT MARK A']
del data_dict['LAY KENNETH L']
del data_dict['SKILLING JEFFREY K']
del data_dict['BHATNAGAR SANJAY']


data=featureFormat(data_dict,features_list,remove_NaN=True)
labels, features = targetFeatureSplit(data)



from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

min_max_scaler = preprocessing.MinMaxScaler()

K_best = SelectKBest(f_classif, k='all')

features_kbest = K_best.fit_transform(features, labels)
feature_scores = ['%.2f' % elem for elem in K_best.scores_ ]
feature_scores_pvalues = ['%.3f' % elem for elem in  K_best.pvalues_ ]
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in K_best.get_support(indices=True)]
print features_selected_tuple

#print features_kbest
from sklearn.cross_validation import train_test_split


features_train, features_test, labels_train, labels_test = train_test_split(features_kbest, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

clf = GaussianNB()
clf.fit(features_train, labels_train)


#from sklearn.model_selection import GridSearchCV. Note able to import GridSearchCV. Used manual screening of paramters. 
#Recall drops significantly if min_samples_split or min_samples_leaf changes from default values.
clf_dt=DecisionTreeClassifier()
#parameters = {'min_samples_split':[2,3,4], 'min_samples_leaf':[1, 2,3,4,5]}
#clf_dt_gs = GridSearchCV(clf_dt, parameters)
clf_dt.fit(features_train, labels_train)

pred = clf_dt.predict(features_test)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

f1 = f1_score(labels_test,pred,average='weighted')
precision=precision_score(labels_test,pred,average='macro')
recall=recall_score(labels_test,pred,average='macro')


import pickle

pickle.dump(data_dict, open("my_dataset.pkl", "wb"))
#Feature scaling is not necessary as we have selected NaiveBayes
pickle.dump(features_list,open("my_feature_list.pkl", "wb"))
pickle.dump(clf,open("my_classifier.pkl", "wb"))









