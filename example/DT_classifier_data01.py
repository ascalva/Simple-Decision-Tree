import pandas as pd
import sys

#
# filename: DT_classifier.py
#
# @author: Alberto Serrano
#
# description: Decision tree generated to predict whether a
#   certain recipe will either produce a muffin or a cupcake.
#

def my_classifier_function(data_record):
    if data_record["Sugar"].values[0] < 18.594285714285714:
        if data_record["Butter or Margarine"].values[0] < 9.5:
            return 0
        else:
            if data_record["Egg"].values[0] < 8.257142857142856:
                if data_record["Baking Powder"].values[0] < 1.2857142857142856:
                    return 0
                else:
                    return 1
            else:
                if data_record["FlourOrOats"].values[0] < 13.492142857142857:
                    return 0
                else:
                    if data_record["FlourOrOats"].values[0] < 41.6:
                        return 1
                    else:
                        return 1
    else:
        if data_record["Egg"].values[0] < 22.728571428571435:
            if data_record["Canned Pumpkin_or_Fruit"].values[0] < 1.8857142857142857:
                return 1
            else:
                return 0
        else:
            return 0


def main():
    path = sys.argv[1]
    data = pd.read_csv(path)
    for i in range(0, data.count()["Type"]):
        data.at[i, "Type"] = my_classifier_function(data.iloc[[i]])
        data.to_csv("DT_classifications.csv", index=False)

main()
