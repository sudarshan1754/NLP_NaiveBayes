###********************************************************************************###
  # __author__ = 'sid'                                                             #
  # This program is written as part of the Natural Language Processing Home Work 4 #
  # @copyright: Sudarshan Sudarshan (Sid)                                          #
###********************************************************************************###

import sys
import os
import math
import nltk

no_of_folds = 10


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
        i = 0  # Starting file in the fold
        group = group_of  # Ending file in the fold

        splits_list = []

        # for each fold split the data
        for each_fold in range(0, folds):
            # list to store the train and test split
            model = []

            # get the test data
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
            i += group_of  # Increment the Starting file in the fold
            group += group_of  # Increment the Ending file in the fold

        return splits_list

    # get the file name list and do cross validation
    def cross_validation(self, files_list, folds):

        no_of_test_files = len(files_list) / folds

        split_data = self.split_file_names(files_list, folds, no_of_test_files)

        return split_data


class NaiveBayes:
    def generate_vocabulary_of_class(self, directory, file_names, vocab):

        for each_file in file_names:
            file_path = directory + "/" + each_file
            file_content = open(file_path, "r")
            for line in file_content.readlines():
                tokens = nltk.WhitespaceTokenizer().tokenize(line)
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = 1
                    else:
                        vocab[token] += 1

        return vocab

    def generate_vocabulary_of_file(self, directory, file_name):

        file_path = directory + "/" + file_name
        file_content = open(file_path, "r")

        vocab = {}
        for line in file_content.readlines():
            tokens = nltk.WhitespaceTokenizer().tokenize(line)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = 1
                else:
                    vocab[token] += 1

        return vocab

    def generate_vocabulary(self, directories, data):

        vocabulary = {}
        for i, each_directory in enumerate(directories):
            self.generate_vocabulary_of_class(each_directory, data[i], vocabulary)

        return vocabulary

    # def read_model_file(self, file_name):
    # pass

    def classify_test(self, test_data, priors, conditional_probability, directories, train_tokens_each_class,
                      vocab_size):

        # test data has pos files in test_data[0]
        # and neg files in test_data[1]
        correct_pos_classified = 0
        correct_neg_classified = 0

        # get the priors
        pos_prior = priors[0]
        neg_prior = priors[1]
        for i, each_test in enumerate(test_data):
            for each_file in each_test:
                score_pos = math.log(pos_prior, 10)
                score_neg = math.log(neg_prior, 10)
                test_file_vocab = self.generate_vocabulary_of_file(directories[i], each_file)
                for word in test_file_vocab:
                    # check for positive files
                    if word not in conditional_probability[0]:
                        new_cp = math.log((1 / float(train_tokens_each_class[0] + vocab_size)), 10)
                        score_pos += new_cp * test_file_vocab[word]
                    else:
                        score_pos += math.log(conditional_probability[0][word], 10) * test_file_vocab[word]

                    # check for negative files
                    if word not in conditional_probability[1]:
                        new_cp = math.log((1 / float(train_tokens_each_class[1] + vocab_size)), 10)
                        score_neg += new_cp * test_file_vocab[word]
                    else:
                        score_neg += math.log(conditional_probability[1][word], 10) * test_file_vocab[word]

                if i == 0:
                    if score_pos >= score_neg:
                        correct_pos_classified += 1
                if i == 1:
                    if score_neg >= score_pos:
                        correct_neg_classified += 1

        return (correct_pos_classified + correct_neg_classified) / float(len(test_data[0]) + len(test_data[1]))

    def naive_bayes(self, directories, data, fold, test_data):

        # generate vocabulary
        vocab = self.generate_vocabulary(directories, data)

        # get the total number of files/docs
        total_files = len(data[0]) + len(data[1])

        # file to write the training results
        model_file = open("model_" + str(fold), "w")

        priors = []
        conditional_prob = []
        total_tokens_in_class = []

        for i, each_class in enumerate(data):
            # get the prior
            priors.append(len(each_class) / float(total_files))

            # get the tokens for that class
            class_tokens = {}
            self.generate_vocabulary_of_class(directories[i], each_class, class_tokens)

            total_tokens = 0
            for token, occurrence in class_tokens.iteritems():
                total_tokens += occurrence
            total_tokens_in_class.append(total_tokens)

            # find the conditional probability
            conditional_prob_of_class = {}
            for each_word in vocab:
                if each_word in class_tokens:
                    conditional_prob_of_class[each_word] = (class_tokens[each_word] + 1) / float(
                        total_tokens + len(vocab))
                else:
                    conditional_prob_of_class[each_word] = 1 / float(total_tokens + len(vocab))

            conditional_prob.append(conditional_prob_of_class)

        model_file.write("priors" + "\t" + str(priors[0]) + "\t" + str(priors[1]) + "\n")

        for each_word in vocab:
            model_file.write(str(each_word) + "\t" + str(conditional_prob[0][each_word]) + "\t"
                             + str(conditional_prob[1][each_word]) + "\n")

        # Do testing
        return self.classify_test(test_data, priors, conditional_prob, directories, total_tokens_in_class, len(vocab))


if __name__ == "__main__":

    print "\n-------------------------WELCOME-------------------------\n"
    pos_dir = raw_input('Enter the directory path which has positive files:')
    neg_dir = raw_input('Enter the directory path which has negative files:')

    if len(pos_dir) == 0 or len(neg_dir) == 0:
        print "Enter the valid path"
        sys.exit()

    print "\n----------------------IMPLEMENTATION---------------------\n"
    directories = [pos_dir, neg_dir]

    pre = PreProcess()

    pos_file_Names = pre.read_file_names(pos_dir)
    neg_file_Names = pre.read_file_names(neg_dir)

    # Structure of pos_split and neg_split:
    # [[[train_fold_1],[test_fold_1]]
    # [[train_fold_2],[test_fold_2]]
    #  [[train_fold_3],[test_fold_3]]
    #  ..............................
    #  ..............................
    #  ..............................]
    pos_split = pre.cross_validation(pos_file_Names, no_of_folds)
    neg_split = pre.cross_validation(neg_file_Names, no_of_folds)

    print "Finished splitting the data for cross validation..."

    nb = NaiveBayes()
    accuracy = []
    for each_fold in range(0, no_of_folds):
        # get the train data
        train_pos_data = pos_split[each_fold][0]
        train_neg_data = neg_split[each_fold][0]
        train_data = [train_pos_data, train_neg_data]

        # get the test data
        test_pos_data = pos_split[each_fold][1]
        test_neg_data = neg_split[each_fold][1]
        test_data = [test_pos_data, test_neg_data]

        fold_accuracy = nb.naive_bayes(directories, train_data, each_fold, test_data)
        accuracy.append(fold_accuracy)
        print "Done with folding: " + str(each_fold + 1)

    print "\n-------------------------RESULTS-------------------------"
    print "Average of " + str(no_of_folds) + "-fold Cross validation: " \
          + str("{0:.2f}".format((sum(accuracy) / len(accuracy)) * 100)) + "%"
    # print accuracy
    print "---------------------------------------------------------"