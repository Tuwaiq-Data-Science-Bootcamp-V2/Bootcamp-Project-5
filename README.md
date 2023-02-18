# Job Placement 


## Introduction
Due to the growing need of educated and talented individuals, especially in developing countries, recruiting fresh graduates is a routine practice for organizations. Conventional recruiting methods and selection processes can be prone to errors and in order to optimize the whole process, some innovative methods are needed.


## Dataset Overview
This [Dataset](https://www.kaggle.com/datasets/ahsan81/job-placement-dataset) contains different attribute of the candidates educational history and work experience.

## Dataset Description
| Column               | Description                                                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| gender | Gender of the candidate  |
| ssc_percentage | Senior secondary exams percentage (10th Grade) |
| ssc_board | Board of education for ssc exams|
| hsc_percentage | Higher secondary exams percentage (12th Grade)|
| hsc_borad  | Board of education for hsc exams |
| hsc_subject  | Subject of study for hsc |
| degree_percentage  | Percentage of marks in undergrad degree |
| undergrad_degree | Undergrad degree majors |
| work_experience  | Past work experience |
| emp_test_percentage | Aptitude test percentage |
| specialization  | SPostgrad degree majors - (MBA specialization) |
| mba_percent   | Percentage of marks in MBA degree |
| status (TARGET) | Status of placement (Placed / Not Placed) |
            


## Machine Learning models

- Decision Tree
From the Decision Tree classifier the confusion_matrix, we have the following observations:
12 TN predictions: zeros predicted correctly.
11 FN predictions: ones wrongly predicted as zeros.
3 FP predictions: zeros that were wrongly predicted as ones.
28 TP predictions: ones predicted correctly.
Accuracy = 74%

- Decision Tree with standard scaler
From the Decision Tree classifier the confusion_matrix, we have the following observations:
13 TN predictions: zeros predicted correctly
10 FN predictions: ones wrongly predicted as zeros
2 FP predictions: zeros that were wrongly predicted as ones
29 TP predictions: ones predicted correctly
Accuracy = 78% 

- Naive Bayes
From the Decision Tree classifier the confusion_matrix, we have the following observations:
4 TN predictions: zeros predicted correctly
0 FN predictions: ones wrongly predicted as zeros
2 FP predictions: zeros that were wrongly predicted as ones
16 TP predictions: ones predicted correctly
Accuracy = 91% 

- Random Forest
From the Decision Tree classifier the confusion_matrix, we have the following observations:
4 TN predictions: zeros predicted correctly
0 FN predictions: ones wrongly predicted as zeros
2 FP predictions: zeros that were wrongly predicted as ones
16 TP predictions: ones predicted correctly
Accuracy = 91%


## Final results
                                                                                               
|  # | Model   | Accuracy %             |
|--- | ------- | ---------------------- |
| 1 | Decision Tree    | 74%  |
| 2 | Decision Tree with standard scaler   | 78%  |
| 3 | XGBoosting xgboost with standard scaler    | 87%  |
| 4 | Naive Bayes    | 91%  |
| 5 | Random Forest  | 91%  |


## Conclusion
In the realm of data science, machine learning algorithms, and model building, the ultimate goal is to build the strongest predictive model while accounting for computational efficiency as well. 
So the project expected not to limit our knowledge to specific machine learning algorithms, we need to be encouraged to go through the additional resources to expand our horizons on machine learning, and to obtain *higher accuracies*.


## Team members
| Team members     | Role                                                                      |
| ---------------- | ------------------------------------------------------------------------- |
| Thekra Alhameedy | Helped in EDA, apply Machine Learning algorithms (Decision Tree, XGBoosting xgboost), and write README markdown file.  |
| Fares Alahmady   | Dataset provider, EDA, creating dashboard, and merge machine learning models.(LEADER) |
| Fares Alshammeri | Helped in EDA, apply Machine Learning algorithms (Random Forests, Naive Bayes), and helped in README markdown file.  |

