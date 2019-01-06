# Decision Tree Trainer #

### Alberto Serrano ###

## Description ##
A simple decision tree trainer that assumes the data being fed is a set of labeled muffin and cupcake recipes. Uses the Gini Index impurity function to make the "best" split. Generates a second file that is able classify the type of recipe.

## Stopping Criteria ##
There are two stopping criterias used in the implementation of the decision tree trainer:
+ If the type of recipes left in the data are more than 90% muffin or cupcake, then stop expanding and add a leaf node.
+ If there are less than 3 recipes left, then stop expanding and insert a leaf node.

## Future work ##
+ Generalize algorithm to accept user defined classes.
+ Make trainer able to create n-class classifier.
+ Use other impurity functions in place of Gini Index.
