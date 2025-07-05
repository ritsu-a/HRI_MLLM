import torch

# motion reconstructions metrics
# 看一看都有什么评价指标?哈哈哈

def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.

    翻译一下:给定两组点云S1,S2,求解最优的s,R,t,使得:
     S1_aligned = s * R @ S1 + t 最接近S2
    即:最小化两组点之间的均方误差
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat, (scale, R, t)

# 每个关节点位置的平均误差
# 这里pred和target都是position
def compute_mpjpe(preds,
                  target,
                  valid_mask=None,
                  pck_joints=None,
                  sample_wise=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance)
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape,
                                              target.shape)  # BxJx3
    mpjpe = torch.norm(preds - target, p=2, dim=-1)  # BxJ

    if pck_joints is None:
        if sample_wise:
            mpjpe_seq = ((mpjpe * valid_mask.float()).sum(-1) /
                         valid_mask.float().sum(-1)
                         if valid_mask is not None else mpjpe.mean(-1))
        else:
            mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_seq
    else:
        mpjpe_pck_seq = mpjpe[:, pck_joints]
        return mpjpe_pck_seq
    
# 其实就是减掉一个基准 从global_pos变成local_pos
def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)

def calc_mpjpe(preds, target, align_inds=[0], sample_wise=True, trans=None):
    # Expects BxJx3
    valid_mask = target[:, :, 0] != -2.0
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    if align_inds is not None:
        preds_aligned = align_by_parts(preds, align_inds=align_inds)
        if trans is not None:
            preds_aligned += trans
        target_aligned = align_by_parts(target, align_inds=align_inds)
    else:
        preds_aligned, target_aligned = preds, target
    mpjpe_each = compute_mpjpe(preds_aligned,
                               target_aligned,
                               valid_mask=valid_mask,
                               sample_wise=sample_wise)
    return mpjpe_each


def calc_pampjpe(preds, target, sample_wise=True, return_transform_mat=False):
    """
    MPJPE没有做点云的配准,这个PAMPJPE有了一步配准,可以补偿掉一些缩放,平移,旋转的误差
    所以理论上PAMPJPE会比MPJPE小一些
    """
    # Expects BxJx3
    target, preds = target.float(), preds.float()
    # extracting the keypoints that all samples have valid annotations
    # valid_mask = (target[:, :, 0] != -2.).sum(0) == len(target)
    # preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(preds[:, valid_mask], target[:, valid_mask])
    # pa_mpjpe_each = compute_mpjpe(preds_tranformed, target[:, valid_mask], sample_wise=sample_wise)

    preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(
        preds, target)
    pa_mpjpe_each = compute_mpjpe(preds_tranformed,
                                  target,
                                  sample_wise=sample_wise)

    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each