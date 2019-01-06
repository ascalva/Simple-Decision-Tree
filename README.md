# Decision Tree Trainer #

### Alberto Serrano ###

## Description ##
A simple decision tree trainer that assumes the data being fed is a set of labeled muffin and cupcake recipes. Uses the Gini Index impurity function to make the "best" split.

## Stopping Criteria ##
There are two stopping criterias used in the implementation of the decision tree trainer:
+ If the type of recipes left in the data are more than 90% muffin or cupcake, then stop expanding and add a leaf node.
+ If there are less than 3 recipes left, then stop expanding and insert a leaf node.
