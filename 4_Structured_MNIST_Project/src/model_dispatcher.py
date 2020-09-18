from sklearn import tree

models = {
    "dt_gini": tree.DecisionTreeClassifier(criterion='gini'),
    "dt_entropy": tree.DecisionTreeClassifier(criterion='entropy')
}