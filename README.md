# Predictive Analytics for Employee Retention using Machine Learning Models

Final year capstone project (BSDA), School of Computing and Artificial Intelligence, Sunway University.

This project predicts employee attrition using the IBM HR Analytics dataset, comparing eight model configurations, six standard algorithms plus two original hybrids, under real class imbalance, and using SHAP to explain what each model is actually picking up on.

## Why this project

Replacing an employee typically costs 50 to 200 percent of their annual salary. Most HR analytics is reactive: it explains turnover after someone has already left. This project tries to flag risk earlier, and to do it in a way that's interpretable rather than a black box. The research behind it had three goals: work out which factors most affect turnover, compare how different algorithms handle a strongly imbalanced dataset, and find a feature selection approach that keeps performance and interpretability both intact.

## Dataset

IBM HR Analytics Employee Attrition dataset: 1,470 employees, 35 features, no missing values. Twenty-six numerical features and nine categorical ones, covering demographics, compensation, tenure and satisfaction. Of the 1,470 employees, 237 (16.1%) left and 1,233 (83.9%) stayed, which is the imbalance the whole pipeline is built around.

The CSV itself isn't in this repo. You can get it from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and place it next to the notebook, or upload it directly when Colab asks.

## Method

Five phases, each tied to one of the three goals above: data exploration, preprocessing and feature engineering, model development, evaluation, and pulling out the insights at the end.

Preprocessing dropped four columns with zero variance, encoded the categorical features, and added five engineered features: TenureRatio, PromotionVelocity, IncomePerYear, WorkLifeRisk and AvgSatisfaction. The data was split 80/20 with stratification, the scaler was fit on the training set only, and SMOTE was applied to the training data (985:191 became 985:985) so the models weren't just learning to predict "stays."

Eight models were trained and compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM and a Neural Network, plus two hybrids. The first uses Random Forest for feature selection and a probability estimate, then feeds both into a Neural Network. The second extracts leaf embeddings from a 200-tree XGBoost model and feeds them into a four-layer deep network.

## Results

All eight models were evaluated on a held-out test set of 294 employees, across seven metrics.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Specificity | G-Mean |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.786 | 0.392 | 0.617 | 0.479 | 0.800 | 0.818 | 0.710 |
| Neural Network | 0.827 | 0.460 | 0.489 | 0.474 | 0.769 | 0.891 | 0.660 |
| GB + Deep Learning | 0.844 | 0.514 | 0.383 | 0.439 | 0.757 | 0.931 | 0.597 |
| XGBoost | 0.861 | 0.625 | 0.319 | 0.423 | 0.786 | 0.964 | 0.555 |
| Random Forest | 0.850 | 0.556 | 0.319 | 0.405 | 0.813 | 0.951 | 0.551 |
| Decision Tree | 0.816 | 0.419 | 0.383 | 0.400 | 0.703 | 0.899 | 0.587 |
| SVM | 0.816 | 0.415 | 0.362 | 0.386 | 0.723 | 0.903 | 0.572 |
| RF + Neural Network | 0.840 | 0.500 | 0.298 | 0.373 | 0.735 | 0.943 | 0.530 |

None of the eight comes out ahead on every metric. Logistic Regression gets the best F1 and recall, mostly because of balanced class weighting. XGBoost is the most precise and the most conservative. Random Forest has the best overall ROC-AUC. Which one you'd actually deploy depends on whether an organisation cares more about catching every possible leaver or avoiding false alarms.

## What the data shows

Overtime is the single strongest signal: 30.5% attrition among employees who work overtime, versus 10.4% for those who don't. Compensation matters too, leavers earn a median of $3,000 to $4,000 a month against $5,500 to $6,500 for people who stay. Attrition is highest in the first two years and drops off sharply after that. Sales Representatives leave at nearly 40%, Research Directors under 5%. Stock options seem to help retention: 24% attrition at option level 0, down to 8% at level 2.

SHAP analysis on XGBoost lines up with this. OverTime, MonthlyIncome, Age, TotalWorkingYears and JobSatisfaction are consistently the top predictors across Decision Tree, Random Forest and XGBoost, and the direction makes sense: overtime pushes toward predicted attrition, higher income pushes the other way. The engineered AvgSatisfaction feature also lands among the top predictors, which was the point of building it in the first place.

## Recommendations

Reduce mandatory overtime where possible. Keep compensation benchmarked against the market. Look at expanding stock option participation, since it's cheaper than raising salaries and seems to help. Put more effort into onboarding in the first two years. Keep an eye on satisfaction scores as an early warning sign rather than only reacting to exit interviews.

## Limitations

The dataset is a single snapshot, so it can't say anything about how risk changes over someone's career. F1 scores across all eight models sit between 0.37 and 0.48, which is moderate rather than strong, so these models are better used to flag people worth a closer look than to make decisions on their own. It's also a synthetic dataset, so it may not fully match how a real company's data would behave. Further work could look at longitudinal data with proper survival analysis, semi-supervised approaches using unlabelled current employees, reinforcement learning for sequencing interventions, and a proper fairness audit of the models themselves.

## Repository structure

```
CP2/
├── CP2.ipynb           # full pipeline: EDA, preprocessing, 8 models, evaluation, SHAP
├── requirements.txt
└── README.md
```

## Getting started

Open `CP2.ipynb` in Google Colab or Jupyter. Upload `WA_FnUseC_HREmployeeAttrition.csv` to the same environment (see Dataset above), then run the cells in order. The notebook installs `xgboost`, `imbalanced-learn` and `shap` itself in the first cell.

To install everything locally instead:

```
pip install -r requirements.txt
```

## Author

Ian Chong Yi Ren (22092902), BSDA, School of Computing and Artificial Intelligence, Sunway University. Supervised by Dr Samuel Mofoluwa Ajibade.

## License

No license has been added yet. If you want others to be able to reuse this, GitHub can generate one from Add file > Create new file in the repo.
