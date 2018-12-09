# iML challenge

## 1. Links

- [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

## 2. Methods tested
### Linear:
 - LinearClassifier : 1.72767
 - Rounded LinearRegression : 0.97520
 - Rounded LinearRegression with gender added to feature: 0.97520
 - Rounded Bagging 10 estimators of linear regression with 0.25 of LS: 0.99946

### DT:
 - DecisionTreeRegressor : 2.47322 (toy_example)
 - DecisionTreeRegressor with gender of user added to feature: 2.45850

### RF:
 - depth = 8; feat = 0.7; est = 50 : 1.15

### kNN
- Bagging 10 estimators of 10-NN with 0.25 of LS (can't more, memory error..): 1.52260
- Rounded Bagging 10 estimators of 10-NN regressor with 0.25 of LS: 1.20338

## 3. Methods to test

1.
2.
3. Neural network
4.

## 4. Remarks
- Please use toy_example as template: create output in 'outputs' dir and keep name of your .py file to create the name of the output file.

- don't feel like adding features about users or movies would improve prediction. See tested 4. and 5.
