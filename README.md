# Car-Price-Analysis.
The dataset is taken from KAGGLE,

Problem Statements: (The questions are taken from Google and HackerRank both combined)
1) Drop all the cars where the price < 10000.
2) Which car was bought maximum number of times by customers and whats the highest price for a car in the dataset?
3) Sort and Return the resultant dataframe in acending order by price.
4) Does Fuel Type affecting the price of cars differently?
5) Which variables are significant in predicting the price of a car.
6) How well those variables describe the price of a car.

The Conclusion of the whole analysis:
1) Data collecting: Gathered all the relevent information about the data which are essential for the analysis.
2) Data Sanitization: Then performed data sanitization(checking null values and duplicated values), luckily there were none present in dataset.
3) Problem Statements: Solved the problem statements first because EDA would messed up my further analysis for ML purposes as outliers were needed to be detected for feeding the model.
4) EDA: Outliers were detected, there were 3 outliers present in the dataset, index no.: 16, 73, 74. after that instead of dropping these values from the original dataset, created a new dataset which doesn't include these values as new_df1. After that plotted a heat map to find the better correlation between the variables, then converted those coulmn values whose values where into strings and converted them into 0's and 1's because ML models cannot read string values. These values were removed('enginetype','fuelsystem','cylindernumber','carbody') because these four columns contains too many different types of strings which is difficult to assign values to, so dropping these columns for further procedure would be better option.
5) Model Building: Dropped 'Carname' and taken the 'price' as Y, trained the dataset in 70:30 ratio. then imported the linear regression and fitted in the model.
6) Prediction: MeanAbsoluteError: 2224.0553273384357, MeanSquaredError: 9088371.645255297 and R2 score: 0.686423340300427
7) Regression Plot(Actual vs Predicted Values): The predicted prices as as close as the actual price till 15000rs but after 15000rs there is difference being observed. most probably because of the 'cylindernumber' column, but as it is a self analysis(done for personal use, mostly skilled based) so there always a room for improvemnets.
