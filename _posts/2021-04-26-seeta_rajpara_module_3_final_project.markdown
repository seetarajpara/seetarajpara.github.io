---
layout: post
title:      "Seeta Rajpara Module 3 Final Project"
date:       2021-04-26 04:03:50 +0000
permalink:  seeta_rajpara_module_3_final_project
---


For this project, I chose to work through the Tanzanian Water Pump Problem.

* Tanzanian Water Pump Problem
* https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/submissions/

### Problem Description: Can you predict which water pumps are faulty?

Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which need some repairs, and which don't work at all? This is an intermediate-level practice competition. Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

The features in this dataset

Your goal is to predict the operating condition of a waterpoint for each record in the dataset. You are provided the following set of information about the waterpoints:

* amount_tsh - Total static head (amount water available to waterpoint)
* date_recorded - The date the row was entered
* funder - Who funded the well
* gps_height - Altitude of the well
* installer - Organization that installed the well
* longitude - GPS coordinate
* latitude - GPS coordinate
* wpt_name - Name of the waterpoint if there is one
* num_private -
* basin - Geographic water basin
* subvillage - Geographic location
* region - Geographic location
* region_code - Geographic location (coded)
* district_code - Geographic location (coded)
* lga - Geographic location
* ward - Geographic location
* population - Population around the well
* public_meeting - True/False
* recorded_by - Group entering this row of data
* scheme_management - Who operates the waterpoint
* scheme_name - Who operates the waterpoint
* permit - If the waterpoint is permitted
* construction_year - Year the waterpoint was constructed
* extraction_type - The kind of extraction the waterpoint uses
* extraction_type_group - The kind of extraction the waterpoint uses
* extraction_type_class - The kind of extraction the waterpoint uses
* management - How the waterpoint is managed
* management_group - How the waterpoint is managed
* payment - What the water costs
* payment_type - What the water costs
* water_quality - The quality of the water
* quality_group - The quality of the water
* quantity - The quantity of water
* quantity_group - The quantity of water
* source - The source of the water
* source_type - The source of the water
* source_class - The source of the water
* waterpoint_type - The kind of waterpoint
* waterpoint_type_group - The kind of waterpoint

#### Distribution of Labels The labels in this dataset are simple. There are three possible values:

* functional - the waterpoint is operational and there are no repairs needed
* functional needs repair - the waterpoint is operational, but needs repairs
* non functional - the waterpoint is not operational

## Data Cleaning and Feature Exploration
- list of variables that might be basically the same (text taken from problem description on competition site):
            
            Geographic location:
            - basin - Geographic water basin
            - subvillage - Geographic location
            - region - Geographic location
            - region_code - Geographic location (coded)
            - district_code - Geographic location (coded)
            - lga - Geographic location
            - ward - Geographic location
            
            Waterpoint operator:
            - scheme_management - Who operates the waterpoint
            - scheme_name - Who operates the waterpoint
            
            Extraction method:
            - extraction_type - The kind of extraction the waterpoint uses
            - extraction_type_group - The kind of extraction the waterpoint uses
            - extraction_type_class - The kind of extraction the waterpoint uses
            
            Waterpoint management:
            - management - How the waterpoint is managed
            - management_group - How the waterpoint is managed
            
            Payment:
            - payment - What the water costs
            - payment_type - What the water costs
            
            Water quality:
            - water_quality - The quality of the water
            - quality_group - The quality of the water
            
            Water quantity:
            - quantity - The quantity of water
            - quantity_group - The quantity of water
            
            Water source:
            - source - The source of the water
            - source_type - The source of the water
            - source_class - The source of the water

            Waterpoint type:
            - waterpoint_type - The kind of waterpoint
            - waterpoint_type_group - The kind of waterpoint
            
- I might not need all these variables since there's probably some overlap and superfluous information that would be fed into an ML model

### Performed in-depth cleaning and feature engineering
- see [notebook](https://github.com/seetarajpara/ds_mod3_project/blob/18db360840e8c2225b17a2b238321174f3ba744b/Seeta%20Rajpara%20Module%203%20Final%20Project%20(2).ipynb) for details

## Modeling Using Random Forest
- From documentation: 
    - A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.
    - Random Forests grows many classification trees. To classify a new object from an input vector, put the input vector down each of the trees in the forest. Each tree gives a classification, and we say the tree "votes" for that class. The forest chooses the classification having the most votes (over all the trees in the forest)

- For this exercise, have to use multinomial logistic regression since there are 3 target labels (functional, non functional, functional need repair)
- Tuning parameters to maximize accuracy:
    - modifying feature engineering techniques
    - updated the parameters for RFC empirically: random_state, min_samples_split, max_features)

### Past models tried with much lower scores:
- Pipeline: 38.56% accuracy
- XGBClassifier: 47.63% accuracy
- RandomForestClassifier: 72-80% accuracy

- After determining best parameters using RandomizedSearchCV, defined random forest classification model:

```
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100,
                                min_samples_split = 20,
                                min_samples_leaf = 1,
                                criterion = 'gini', 
                                max_features = 10, 
                                oob_score = True, 
                                random_state=250)
																```

- Tested this on training dataset:

```
def train_eval(X,y, clf):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=250)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    scaler = StandardScaler()
    
    X_train_clf = scaler.fit_transform(X_train)
    X_test_clf = scaler.transform(X_test)
    
    clf.fit(X_train_clf, y_train)
    
    y_pred_train = clf.predict(X_train_clf)
    y_pred = clf.predict(X_test_clf)
    
    return accuracy_score(y_train,y_pred_train), accuracy_score(y_test, y_pred)```

```
%%time

train_eval(X_train, y_train, clf_rf)
```

> (47520, 20) (11880, 20) (47520,) (11880,)
CPU times: user 15.8 s, sys: 220 ms, total: 16 s
Wall time: 16.7 s
(0.8419402356902357, 0.727020202020202)

- The accuracy on this training set was about 84%
- Testing using the robust scaler showed a slight increase in performance, which is what I used to submit to the competition

```
def predictor_model(X_train, X_test, y_train, clf):
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score
    
    y_pred = pd.DataFrame()
    temp_test = X_test.reset_index()
    y_id = temp_test['id']
    
    scaler = RobustScaler()
    
    X_train_clf = scaler.fit_transform(X_train)
    X_test_clf = scaler.transform(X_test)
    
    clf.fit(X_train_clf, y_train)
    y_pred_train = clf.predict(X_train_clf)
    
    print(f'\nThe accuracy score of the training set is {round(accuracy_score(y_train, y_pred_train), 5)}\n')
    
    prediction = pd.DataFrame(clf.predict(X_test_clf))
    y_pred = pd.concat([y_id, prediction], axis='columns')
    y_pred.rename(columns={0:'status_group'}, inplace=True)
    y_pred.set_index('id', inplace=True)
    
    return y_pred
```


```
%%time

df = predictor_model(X_train, X_test, y_train, clf_rf)
```

> The accuracy score of the training set is 0.84438

CPU times: user 19.7 s, sys: 286 ms, total: 19.9 s
Wall time: 20.8 s

- the accuracy here is slightly higher, so I submitted this and got around 80% accuracy on the competition site

## Data Visualizations
### Mapped waterpoints by functionality - color coded by construction_year and sized by population:


https://github.com/seetarajpara/ds_mod3_project/blob/main/output_150_1.png

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_151_1.png

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_152_1.png


### Boxplots illustrating relationship between categorical features and functionality:

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_157_1.png

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_158_1.png

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_159_1.png

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_160_1.png

### Barplots illustrating relationship between categorical features and functionality:

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_161_1.png

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_167_1.png

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_168_1.png

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_169_1.png

### Pearson correlation heatmap illustrating relationship between numerical features and functionality (encoded):

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_165_1.png

### Pairplot showing relationships between numerical features and functionality (encoded):

https://github.com/seetarajpara/ds_mod3_project/blob/main/output_171_1.png

## Conclusions 
- Overall, the accuracy of the resulting data frame was relatively high
- The frequency percentage of each functionality label was empirically determined on the competition site
## Next Steps
- Our model can always be improved upon, and further optimizing the random forest model parameters can help generate higher accuracy scores
- In addition, some of the data cleaning steps could be fine tuned further, perhaps exploring in more detail, each variable’s influence on the target class for functionality
- Another recommendation: generate more visualizations of these engineered features to qualitatively determine whether they are influential to the final classification# Enter your title here

The content of your blog post goes here.
