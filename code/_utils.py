import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_hits(predictions, labels):
  # if predictions.shape != labels.shape:
  #     print('prediction shape and label are not the same')
  #     exit()
  predictions = np.squeeze(predictions)
  labels = np.squeeze(labels)
  if predictions.ndim != 1:
    print('predictions are not a 1-D vector')
    exit()
  if labels.ndim != 1:
    print('labels are not a 1-D vector')
    exit()

  pos_label_mask = (labels == 1)
  neg_label_mask = (labels == 0)
  training_hits = labels == predictions
  pos_pred_hits_sum = np.sum(np.logical_and(training_hits, pos_label_mask))
  neg_pred_hits_sum = np.sum(np.logical_and(training_hits, neg_label_mask))
  training_hits_sum = np.sum(training_hits)

  num_training_points = training_hits.shape[0]
  num_pos_points = np.sum(pos_label_mask)
  num_neg_points = np.sum(neg_label_mask)

  return training_hits_sum, pos_pred_hits_sum, neg_pred_hits_sum, num_training_points, num_pos_points, num_neg_points


# pred = np.array([1,0,1,0])[np.newaxis,:]
# labels = np.array([0,0,1,0])
# all_hits, pos_hits, neg_hits, points, pos_points, neg_points = compute_hits(pred, labels)
#
# print('finished')

# import torch
# import torch.nn.functional as F
#
# # input is of size N x C = 3 x 5
# input = torch.randn(3, 2, requires_grad=True)
# # each element in target has to have 0 <= value < C
# target = torch.tensor([1, 0, 1])
# # out = F.log_softmax(input, dim=1)
# # output = F.nll_loss(out, target)
# output = F.cross_entropy(input, target)
# output.backward()





class LabelSmoothingCrossEntropy(nn.Module):
  def __init__(self, epsilon: float = 0.1, reduction='mean'):
    super().__init__()
    self.epsilon = epsilon
    self.reduction = reduction

  def forward(self, preds, target):
    n = preds.size()[-1]
    log_preds = F.log_softmax(preds, dim=-1)
    loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
    nll = F.nll_loss(log_preds, target, reduction=self.reduction)
    return self.linear_combination(loss / n, nll, self.epsilon)

  def linear_combination(self, x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

  def reduce_loss(self, loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss