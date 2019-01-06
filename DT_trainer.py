import pandas as pd
import numpy as np
import sys

#
# filename: DT_classifier.py
#
# @author: Alberto Serrano
#
# usage: python3 [training program] [training data]
#
# description: Uses a set of training data supplied to generate
#              a decision tree.
#

def gini(p_muffin, p_cupcake):
    """
    Calculates the gini index using the probability of finding a
    muffin and a cupcake from a given region in the data.
    """
    return 1 - (p_muffin ** 2) - (p_cupcake ** 2)


def recipeType(data, catagory):
    """
    Using supplied data, recipeType() finds and returns a data frame for
    recipes that are muffins and recipes that are cupcakes.
    """
    muffin  = data[data["Type"] == "Muffin" ][catagory]
    cupcake = data[data["Type"] == "cupcake"][catagory]

    return muffin, cupcake


def emit_header():
    """
    Writes out all necessary libraries required by the program and the header.
    """
    header = "import pandas as pd\n" \
           + "import sys\n\n" \
           + "#\n" \
           + "# filename: DT_classifier.py\n" \
           + "#\n" \
           + "# @author: Alberto Serrano\n" \
           + "#\n" \
           + "# description: Decision tree generated to predict whether a\n" \
           + "#   certain recipe will either produce a muffin or a cupcake.\n" \
           + "#\n\n"

    file_pointer.write(header.replace("\t", "    "))


def printDecision(best_catagory, best_thresh, tabs, if_else):
    """
    Based on the value of the if_else parameter, print either an if
    statement or an else statement.
    If the condition to print is an if statement, use best_thresh and
    best_catagory to write it.
    """
    tab_string = "".join(["\t"] * tabs).replace("\t", "    ")
    stmt       = ""

    if if_else:
        stmt += "else:\n"

    else:
        stmt += "if data_record[\"{0}\"].values[0] < {1}:\n" \
               .format(best_catagory, best_thresh)

    file_pointer.write(tab_string + stmt)


def printStatement(leaf, tabs):
    """
    Using the leaf parameter, which is the statement to be printed and the
    number of tabs, write statement to the file.
    """
    tab_string = "".join(["\t"] * tabs).replace("\t", "    ")

    file_pointer.write(tab_string + leaf + "\n")


def resolveLeaf(data, tabs):
    """
    If more than 50%  of the data is either a muffin or cupcake, generate a
    return statement for the type of recipe it was determined to be.
    """
    if (data[data["Type"] == "Muffin"]["Type"].count() / data["Type"].count()) > 0.5:
        printStatement("return 0", tabs) # return muffin

    else:
        printStatement("return 1", tabs) # return cupcake


def buildTree(data, tabs):
    """
    Train the classifier by splitting all the values in a catagory using
    different thresholds to find the smallest weighted Gini value. Once the
    data is split, recurse on each partition.
    """
    best_weight   = sys.maxsize
    best_thresh   = 0
    best_catagory = ""

    # Stopping criterias:
    # If more than 90% of the recipes are muffins, stop
    if (data[data["Type"] == "Muffin"]["Type"].count() / data["Type"].count()) > 0.9:
        printStatement("return 0", tabs) # return muffin
        return

    # If more than 90% of the recipes are cupcakes, stop
    elif (data[data["Type"] == "Cupcake"]["Type"].count() / data["Type"].count()) > 0.9:
        printStatement("return 1", tabs) # return 1
        return

    # If there are less than 3 values left in the data, stop expanding
    elif data["Type"].count() <= 3:
        resolveLeaf(data, tabs)
        return

    # Expand on each catagory
    for catagory in data:

        # Do not use the type as an attribute to train on
        if catagory == "Type":
            continue

        # Generate thresholds
        thresh = np.linspace(data[catagory].min(), data[catagory].max(), 15)

        for t in thresh:
            data_under = data[data[catagory] < t]
            muffin_under, cupcake_under = recipeType(data_under, catagory)

            data_over  = data[data[catagory] >= t]
            muffin_over, cupcake_over   = recipeType(data_over, catagory)

            cost_under = 0.0
            cost_over = 0.0

            # Make sure not dividing by zero
            if data_under[catagory].count() != 0:
                p_muffin   = muffin_under.count() / data_under.count()[catagory]
                p_cupcake  = cupcake_under.count() / data_under.count()[catagory]
                cost_under = gini(p_muffin, p_cupcake)

            if data_over[catagory].count() != 0:
                p_muffin   = muffin_over.count() / data_over.count()[catagory]
                p_cupcake  = cupcake_over.count() / data_over.count()[catagory]
                cost_over  = gini(p_muffin, p_cupcake)

            # Calculate the weighted average
            n_under  = data_under[catagory].count() / data[catagory].count()
            n_over   = data_over[catagory].count() / data[catagory].count()
            weighted = (n_under * cost_under) + (n_over * cost_over)

            # Record the smallest weighted Gini index
            if weighted < best_weight:
                best_weight   = weighted
                best_thresh   = t
                best_catagory = catagory

    # Recurse on each partition of the data and print the appropriate
    # conditional statement to the clasifier file.
    printDecision(best_catagory, best_thresh, tabs, 0)
    buildTree(data[data[best_catagory] <  best_thresh], tabs + 1)

    printDecision(best_catagory, best_thresh, tabs, 1)
    buildTree(data[data[best_catagory] >= best_thresh], tabs + 1)


def emit_decision_tree(data):
    """
    Print all the parts of the decision tree.
    """
    dt_f = "def my_classifier_function(data_record):\n" #\
    file_pointer.write(dt_f.replace("\t", "    "))

    buildTree(data, 1)

    file_pointer.write("\n\n")


def emit_calls():
    """
    Generate the end of the classifier function, this takes care of reading in
    the csv file and passing one recipe at a time into the classifier function
    and updates the type of recipe. Then outputs the updated data to a new file.
    Also calls on itself.
    """
    main_f = "def main():\n" \
           + "\tpath = sys.argv[1]\n" \
           + "\tdata = pd.read_csv(path)\n" \
           + "\tfor i in range(0, data.count()[\"Type\"]):\n" \
           + "\t\tdata.at[i, \"Type\"] = my_classifier_function(data.iloc[[i]])\n" \
           + "\t\tdata.to_csv(\"HW_05_Serrano_Alberto_MyClassifications.csv\", index=False)\n\n" \
           + "main()\n"

    file_pointer.write(main_f.replace("\t", "    "))


def main():
    # Makes my life easier by making the file pointer into a global data
    global file_pointer

    if len(sys.argv) != 2:
        print("Usage: python3 [trainer program] [training data]")
        return

    path_data       = sys.argv[1]
    path_classifier = "HW_05_Serrano_Alberto_Classifier.py"
    train_data      = pd.read_csv(path_data)
    file_pointer    = open(path_classifier, "w")

    # Write the decision tree and emit all the contents of the program so that
    # it is runnable.
    emit_header()
    emit_decision_tree(train_data)
    emit_calls()

    file_pointer.close()

main()
