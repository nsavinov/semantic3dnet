#!/usr/bin/env python

class ConfusionMatrix:
  """Streaming interface to allow for any source of predictions. Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""
  def __init__(self, number_of_labels):
    self.number_of_labels = number_of_labels
    self.confusion_matrix = []
    for i in xrange(number_of_labels):
      self.confusion_matrix.append([0] * number_of_labels)
  """labels are integers from 0 to number_of_labels-1"""
  def count_predicted(self, ground_truth, predicted, number_of_added_elements=1):
    self.confusion_matrix[ground_truth][predicted] += number_of_added_elements
  """labels are integers from 0 to number_of_labels-1"""
  def get_count(self, ground_truth, predicted):
    return self.confusion_matrix[ground_truth][predicted]
  """returns list of lists of integers; use it as result[ground_truth][predicted]
     to know how many samples of class ground_truth were reported as class predicted"""
  def get_confusion_matrix(self):
    return self.confusion_matrix
  """returns list of 64-bit floats"""
  def get_intersection_union_per_class(self):
    matrix_diagonal = [self.confusion_matrix[i][i] for i in xrange(self.number_of_labels)]
    errors_summed_by_row = [0] * self.number_of_labels
    for row in xrange(self.number_of_labels):
      for column in xrange(self.number_of_labels):
        if row != column:
          errors_summed_by_row[row] += self.confusion_matrix[row][column]
    errors_summed_by_column = [0] * self.number_of_labels
    for column in xrange(self.number_of_labels):
      for row in xrange(self.number_of_labels):
        if row != column:
          errors_summed_by_column[column] += self.confusion_matrix[row][column]
    divisor = [matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
               for i in xrange(self.number_of_labels)]
    return [float(matrix_diagonal[i]) / divisor[i] for i in xrange(self.number_of_labels)]
  """returns 64-bit float"""
  def get_average_intersection_union(self):
    values = self.get_intersection_union_per_class()
    return sum(values) / len(values)

def test_confusion_matrix_class():
  real_matrix = [[244310, 10956, 1365, 1003, 9691],
                 [249, 3467, 34, 13, 31],
                 [324, 86, 6474, 19, 1030],
                 [10954, 24223, 924, 897503, 542],
                 [841, 1531, 12339, 38, 96363]]
  number_of_classes = 5
  confusion_matrix = ConfusionMatrix(5)
  for row in xrange(number_of_classes):
    for column in xrange(number_of_classes):
      confusion_matrix.count_predicted(row, column, real_matrix[row][column])
  matrix_to_test = confusion_matrix.get_confusion_matrix()
  # print matrix_to_test
  for row in xrange(number_of_classes):
    for column in xrange(number_of_classes):
      assert matrix_to_test[row][column] == real_matrix[row][column]
  IoU = confusion_matrix.get_intersection_union_per_class()
  # print IoU
  IoU_ground_truth = [0.8735, 0.0854, 0.2865, 0.9597, 0.7872]
  for i in xrange(number_of_classes):
    assert abs(IoU[i] - IoU_ground_truth[i]) < 0.0001
  average_IoU = confusion_matrix.get_average_intersection_union()
  # print average_IoU
  average_IoU_ground_truth = 0.59846888780594
  assert abs(average_IoU - average_IoU_ground_truth) < 0.00000001
  print 'all tests passed successfully!'

# test_confusion_matrix_class()

import sys
import itertools

def main():
  result = ConfusionMatrix(8)
  with open(sys.argv[1]) as gt_file:
    with open(sys.argv[2]) as prediction_file:
      for gt, prediction in itertools.izip(gt_file, prediction_file):
        gt = int(gt)
        prediction = int(prediction)
        if gt > 0: 
          result.count_predicted(gt - 1, prediction - 1)
  print(result.get_intersection_union_per_class())
  print(result.get_average_intersection_union())

if __name__ == "__main__":
  main()
