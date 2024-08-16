from typing import Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    logit_scale,
    lse_square_scale,
    ignore_index,
    n_cols,
    n_rows,
    logits_row_stride,
    labels_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    labels_ptr = labels_ptr + row_idx * labels_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32) * logit_scale
    labels = tl.load(labels_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0).to(tl.float32)
    max_logits = tl.max(logits, 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    loss = tl.sum(labels * (lse - logits), 0)
    z_loss = lse_square_scale * lse * lse
    loss += z_loss
    # Apply ignore_index
    is_ignored = tl.sum(labels, 0) == 0  # Assuming ignored labels are all zeros
    loss = tl.where(is_ignored, 0.0, loss)
    z_loss = tl.where(is_ignored, 0.0, z_loss)
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)

@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    logit_scale,
    lse_square_scale,
    ignore_index,
    n_cols,
    logits_row_stride,
    dlogits_row_stride,
    labels_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride
    labels_ptr = labels_ptr + row_idx * labels_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32) * logit_scale
    labels = tl.load(labels_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0).to(tl.float32)
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    dlogits = (probs - labels) * dloss * logit_scale
    # Apply ignore_index
    is_ignored = tl.sum(labels, 0) == 0  # Assuming ignored labels are all zeros
    dlogits = tl.where(is_ignored, 0.0, dlogits)
    tl.store(dlogits_ptr + col_offsets, dlogits, mask=col_offsets < n_cols)

class CrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_scale=1.0, lse_square_scale=0.0, inplace_backward=False, ignore_index=-100):
        print(logits.shape)
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows, n_cols)

        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        if labels.stride(-1) != 1:
            labels = labels.contiguous()

        MAX_BLOCK_SIZE = 64 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else (16 if BLOCK_SIZE < 128 * 1024 else 32))
        n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
        losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)

        with torch.cuda.device(logits.device.index):
            cross_entropy_fwd_kernel[(n_rows, n_splits)](
                losses,
                lse,
                z_losses,
                logits,
                labels,
                logit_scale,
                lse_square_scale,
                ignore_index,
                n_cols,
                n_rows,
                logits.stride(0),
                labels.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        if n_splits > 1:
            losses = losses.sum(dim=0)
            z_losses = z_losses.sum(dim=0)

        ctx.save_for_backward(logits, lse, labels)
        ctx.mark_non_differentiable(z_losses)
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.inplace_backward = inplace_backward
        ctx.ignore_index = ignore_index

        return losses, z_losses

    @staticmethod
    def backward(ctx, grad_losses, grad_z_losses):
        del grad_z_losses  # z_losses are only for logging.

        logits, lse, labels = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
        def grid(META): return (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))

        with torch.cuda.device(logits.device.index):
            cross_entropy_bwd_kernel[grid](
                dlogits,
                grad_losses,
                logits,
                lse,
                labels,
                ctx.logit_scale,
                ctx.lse_square_scale,
                ctx.ignore_index,
                n_cols,
                logits.stride(0),
                dlogits.stride(0),
                labels.stride(0),
                grad_losses.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        return dlogits, None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    inplace_backward: bool = False,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return CrossEntropyLossFunction.apply(
        logits,
        labels,
        logit_scale,
        lse_square_scale,
        inplace_backward,
        ignore_index,
    )

class FusedSoftCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        logit_scale=1.0,
        lse_square_scale=0.0,
        inplace_backward=False,
        return_z_loss=False,
        ignore_index=-100,
    ):
        super().__init__()
        if reduction not in ["mean", "none", "sum"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none' or 'sum'")
        self.reduction = reduction
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.inplace_backward = inplace_backward
        self.return_z_loss = return_z_loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        assert input.is_cuda and target.is_cuda, "Only support CUDA tensors"
        loss, z_loss = cross_entropy_loss(
            input,
            target,
            logit_scale=self.logit_scale,
            lse_square_scale=self.lse_square_scale,
            inplace_backward=self.inplace_backward,
            ignore_index=self.ignore_index,
        )
        if self.reduction == "mean":
            loss = loss.sum() / (target != self.ignore_index).sum()
            z_loss = z_loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
            z_loss = z_loss.sum()

        if not self.return_z_loss:
            return loss
        return loss, z_loss