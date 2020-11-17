# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary of the problem statement
###### **1) Summary**
Based on a dataset provided which contains customer's information such as age, marital status, education, current loans, housing between others, we seek to predict whether the customers may be interested to take products offered to them accross marketing campaigns.
    
###### **2) How the problem was solved**
The main approach is to design and to execute two results using the same dataset; the first one will be based on the optimization of hyperparameters using a tool on Microsoft Azure ML Studio called HyperDrive, the second one will be run using MIcrosoft Azure AutoML approach; at the end the analyst may compare both results as well as their metrics to take the best decision.

The best performing model was a 0.91836 using VotingEnsemble algorithm, this one was calculated using the autoML Experiment.

## Scikit-learn Pipeline
###### **1) Pipeline Technical details**
1) Architecture: Virtual Machine General Purpose CPU Cluster Compute D-Series V2
2) Data: CSV Format, 21 columns, 32,950 data rows. The is loaded using a TabularDatasetFactory class,  to acqurate the result the is cleaned using the function “clean_data” which is part of the script train.py
3) Classification algorithm:  We use a Scikit-learn Logistic Regression Model with a parameter sampler
4) Hyperparameters: “C” which is the regularization parameter, “max-iter” which define the maximum number of iterations allowed


###### **2) How to choose a parameter sampler**
One of the objective is to obtain different results, in that order we need to optimize the use of the resources, such processing and compute time consumption, that's why one of the best option is to use a RandomParameterSampling class just like in the above example:
    ps = RandomParameterSampling(
        {
            '--C': uniform(0.05, 1),
            '--max_iter': choice(20, 40, 60, 80, 100)
        }
    )

###### **3) How to choose an early stopping policy**
For acqurate results is a good option to set up an early stopping policy, this is parameter of tolerance in our HyperDriveConfig, the idea is to stop unefficient runs which won't reach improving results, then the compute resources will be available for the next experiments. This is an example of how to set up a policy:

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

## AutoML
AutoML is a powerful feature of Machine Learning Studio, comparing with Hyperdrive, the capability to test with different models in almost the half of time is simply outstanding, on each model we can obtain very detailed metrics like the next ones:

## Pipeline comparison
The best result obtained by HyperDrive Experiment was 0.9150227617602428, the experiment based on AutoML was 0.91836 using VotinEnsemble as the best algorithm, in quantitative terms both results are practically the same but the AutoML experiment is more accurate because the value selected was the best one between a series of models tested. Using HyperDrive to test different models will be more dificult because each requires specific pipeline parameters.

## Future work
In particular, I would like to try different models to compare the results, I would like to try the Area Under the Curve metric -even though I'm not sure whether it fits with this dataset, however, it would be interesting to contrast a different approach of analysis.

The selection of the column as the parameter for “label_column_name” may be another area of improvement, in the exercise this column is already given but it won't on a different dataset, it would be nice to apply more criterias on the selection of the appropiate column.

## Proof of cluster clean up
Once the experiments are finished the next line delete the cluster which is not necessary anymore:
aml_compute.delete()
![ccluster_delete.png](./images/ccluster_delete.png?raw=true "Computer Cluster deleted")
