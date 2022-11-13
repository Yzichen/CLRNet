import torch


def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred/N_pos, S=72)     S: absolute x coords
        target: ground truth, shape: (num_target/N_pos, S=72)     S: absolute x coords
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    Returns:
        iou: (num_pred, num_target) /  (N_pos, )
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target   # (N_pos, S)
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)     # (N_pos, S)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)   # (N_pos, S)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)    # (num_pred, num_target, S)
        # (num_pred, num_target, S)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))      # (num_pred, num_target, S)
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))    # (num_pred, num_target, S)

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)    # (num_pred, num_target, S)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)      # (num_pred, num_target)
    return iou


def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length)).mean()