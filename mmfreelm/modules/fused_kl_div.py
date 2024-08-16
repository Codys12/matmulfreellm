import torch
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
    mask = col_offsets < n_cols
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32) * logit_scale
    labels = tl.load(labels_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    max_logits = tl.max(logits, 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + row_idx, lse)
    
    loss = -tl.sum(labels * (logits - lse), 0)
    z_loss = lse_square_scale * lse * lse
    loss += z_loss
    
    # Check if all labels in this row are equal to ignore_index
    is_ignored = tl.all(labels == ignore_index, 0)
    loss = tl.where(is_ignored, 0.0, loss)
    z_loss = tl.where(is_ignored, 0.0, z_loss)
    
    tl.store(loss_ptr + row_idx, loss)
    tl.store(z_loss_ptr + row_idx, z_loss)

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
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride
    labels_ptr = labels_ptr + row_idx * labels_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dloss = tl.load(dloss_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32) * logit_scale
    labels = tl.load(labels_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    lse = tl.load(lse_ptr + row_idx)
    
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    dlogits = (probs - labels) * (dloss * logit_scale)
    
    # Zero out gradients for ignored indices
    is_ignored = tl.all(labels == ignore_index, 0)
    dlogits = tl.where(is_ignored, 0.0, dlogits)
    
    tl.store(dlogits_ptr + col_offsets, dlogits, mask=mask)

class CrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_scale=1.0, lse_square_scale=0.0, ignore_index=-100, inplace_backward=False):
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows, n_cols), "Labels must have the same shape as logits"
        
        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        if labels.stride(-1) != 1:
            labels = labels.contiguous()
        
        MAX_BLOCK_SIZE = 4 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
        
        losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        z_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        
        grid = (n_rows,)
        
        cross_entropy_fwd_kernel[grid](
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
        
        ctx.save_for_backward(logits, lse, labels)
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignore_index = ignore_index
        ctx.inplace_backward = inplace_backward
        
        return losses, z_losses

    @staticmethod
    def backward(ctx, grad_losses, grad_z_losses):
        logits, lse, labels = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
        grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
        
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
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return dlogits, None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index: int = -100,
    inplace_backward: bool = False,
) -> torch.Tensor:
    return CrossEntropyLossFunction.apply(
        logits,
        labels,
        logit_scale,
        lse_square_scale,
        ignore_index,
        inplace_backward,
    )

class FusedSoftCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        reduction="mean",
        logit_scale=1.0,
        lse_square_scale=0.0,
        ignore_index=-100,
        inplace_backward=False,
        return_z_loss=False,
    ):
        super().__init__()
        self.reduction = reduction
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.ignore_index = ignore_index
        self.inplace_backward = inplace_backward
        self.return_z_loss = return_z_loss

    def forward(self, input, target):
        loss, z_loss = cross_entropy_loss(
            input,
            target,
            logit_scale=self.logit_scale,
            lse_square_scale=self.lse_square_scale,
            ignore_index=self.ignore_index,
            inplace_backward=self.inplace_backward,
        )
        
        if self.reduction == "mean":
            valid_elements = (target != self.ignore_index).any(dim=1).sum()
            loss = loss.sum() / valid_elements
            z_loss = z_loss.sum() / valid_elements
        elif self.reduction == "sum":
            loss = loss.sum()
            z_loss = z_loss.sum()
        
        if self.return_z_loss:
            return loss, z_loss
        else:
            return loss