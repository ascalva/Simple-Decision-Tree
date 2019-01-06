import pandas as pd
import numpy as np
import sys
from DT_trainer import open_file_pointer, close_file_pointer, emit_calls, emit_header, emit_decision_tree

#
# filename: DT.py
#
# @author: Alberto Serrano
#
# usage: python3 [training program] [training data]
#
# description: Uses a set of training data supplied to generate
#              a decision tree.
#

def main():

    if len(sys.argv) != 2:
        print("Usage: python3 [trainer program] [training data]")
        return

    path_data       = sys.argv[1]
    path_classifier = "DT_classifier.py"
    train_data      = pd.read_csv(path_data)

    open_file_pointer(path_classifier)

    # Write the decision tree and emit all the contents of the program so that
    # it is runnable.
    emit_header()
    emit_decision_tree(train_data)
    emit_calls()

    close_file_pointer()


if __name__ == '__main__':
    main()
