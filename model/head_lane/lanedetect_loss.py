import torch
import torch.nn.functional as F


def find_k_th_small_in_a_tensor(target_tensor, k_th):
    """Like name, this function will return the k the of the tensor."""
    val, idxes = torch.topk(target_tensor, k=k_th, largest=False)
    return val[-1]


def huber_fun(x):
    """Implement of hunber function."""
    absx = torch.abs(x)
    r = torch.where(absx < 1, x * x / 2, absx - 0.5)
    return r


def cal_loss_cls(cls_targets, cls_preds, NEGATIVE_RATIO = 15, ALPHA = 10):
    # ---------------------------------------------------#
    #  计算mask
    # ---------------------------------------------------#
    cls_targets = cls_targets[..., 1].view(-1)
    pmask = cls_targets > 0
    nmask = ~pmask
    fpmask = pmask.float()
    fnmask = nmask.float()

    # ---------------------------------------------------#
    #  处理
    # ---------------------------------------------------#
    cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
    total_postive_num = torch.sum(fpmask)
    total_negative_num = torch.sum(fnmask)  # Number of negative entries to select
    negative_num = torch.clamp(total_postive_num * NEGATIVE_RATIO, max=total_negative_num, min=1).int()
    positive_num = torch.clamp(total_postive_num, min=1).int()

    # ---------------------------------------------------#
    #  类别损失
    # ---------------------------------------------------#
    bg_fg_predict = F.log_softmax(cls_preds, dim=-1)
    fg_predict = bg_fg_predict[..., 1]
    bg_predict = bg_fg_predict[..., 0]
    max_hard_pred = find_k_th_small_in_a_tensor(bg_predict[nmask].detach(), negative_num)
    fnmask_ohem = (bg_predict <= max_hard_pred).float() * nmask.float()
    total_cross_pos = -torch.sum(ALPHA * fg_predict * fpmask)
    total_cross_neg = -torch.sum(ALPHA * bg_predict * fnmask_ohem)

    # ---------------------------------------------------#
    #  取平均
    # ---------------------------------------------------#
    total_cross_pos = total_cross_pos / positive_num
    total_cross_neg = total_cross_neg / positive_num

    return total_cross_pos, total_cross_neg, pmask, positive_num


def cal_loss_regress(pmask, positive_num, loc_targets,loc_preds, ALPHA = 10, points_per_line = 160):
    #---------------------------------------------------#
    #  计算回归损失
    #---------------------------------------------------#
    loc_preds = loc_preds.view(-1, loc_preds.shape[-1])
    loc_targets = loc_targets.view(-1, loc_targets.shape[-1])
    length_weighted_mask = torch.ones_like(loc_targets)

    length_weighted_mask[..., points_per_line + 1] = ALPHA
    length_weighted_mask[..., points_per_line] = ALPHA

    valid_lines_mask = pmask.unsqueeze(-1).expand_as(loc_targets)
    valid_points_mask = (loc_targets != 0)
    unified_mask = length_weighted_mask.float() * valid_lines_mask.float() * valid_points_mask.float()
    smooth_huber = huber_fun(loc_preds - loc_targets) * unified_mask
    loc_smooth_l1_loss = torch.sum(smooth_huber, -1)
    point_num_per_gt_anchor = torch.sum(valid_points_mask.float(), -1).clamp(min=1)
    total_loc = torch.sum(loc_smooth_l1_loss / point_num_per_gt_anchor)

    total_loc = total_loc / positive_num

    return total_loc
