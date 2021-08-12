import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def iou_joint_loss(pred, target, beta=2.0):
    """
    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for IJL must be a tuple of two elements, 
    including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target
    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] * pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * (1 - (scale_factor + 1).log()).pow(beta)
    loss = loss.sum(dim=1, keepdim=False)
    return loss


@LOSSES.register_module()
class IoUJointLoss(nn.Module):
    """
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(IoUJointLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid is supported.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * iou_joint_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls