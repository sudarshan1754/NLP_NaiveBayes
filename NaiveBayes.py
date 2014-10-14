###********************************************************************************###
  # __author__ = 'sid'                                                             #
  # This program is written as part of the Natural Language Processing Home Work 4 #
  # @copyright: Sudarshan Sudarshan (Sid)                                          #
###********************************************************************************###

import sys
import os


class PreProcess:

    @staticmethod
    def read_file_names(directory_path):

        files = []
        for each_file in os.listdir(directory_path):
            files.append(each_file)

        return sorted(files)

    # split the file names
    @staticmethod
    def split_file_names(file_list, folds, group_of):

        # print no_of_test_files, no_of_train_files
        i = 0                                         # Starting file in the fold
        group = group_of                              # Ending file in the fold

        splits_list = []

        # for each fold split the data
        for each_fold in range(0, folds):
            # list to store the train and test split
            model = []

            #get the test data
            test = []
            for each in range(i, group):
                test.append(file_list[each])

            # get the training data
            train = [x for x in file_list if x not in test]

            # store train and test in a list
            model.append(train)
            model.append(test)

            #store the fold in the list
            splits_list.append(model)

            #increment to next fold
            i += group_of                               # Increment the Starting file in the fold
            group += group_of                           # Increment the Ending file in the fold

        return splits_list

    # get the file name list and do cross validation
    def cross_validation(self, files_list, folds):

        no_of_test_files = len(files_list) / folds

        split_data = self.split_file_names(files_list, folds, no_of_test_files)

        cross_file = open("cross", "w")
        for each in split_data:
            # print each
            cross_file.write(str(each) + "\n")
        return split_data


if __name__ == "__main__":
    pos_dir = raw_input('Enter the directory path which has positive files:')
    neg_dir = raw_input('Enter the directory path which has negative files:')

    if len(pos_dir) == 0 or len(neg_dir) == 0:
        print "Enter the valid path"
        sys.exit()

    pre = PreProcess()

    pos_file_Names = pre.read_file_names(pos_dir)
    neg_file_Names = pre.read_file_names(neg_dir)

    no_of_folds = 10

    # Structure of pos_split and neg_split:
    # [[[train_fold_1],[test_fold_1]]
    #  [[train_fold_2],[test_fold_2]]
    #  [[train_fold_3],[test_fold_3]]
    #  ..............................
    #  ..............................
    #  ..............................]
    pos_split = pre.cross_validation(pos_file_Names, no_of_folds)
    neg_split = pre.cross_validation(neg_file_Names, no_of_folds)


    # print len(pos_split)
    # print len(neg_split)


