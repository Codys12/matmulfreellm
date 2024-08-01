import torch
import triton
import triton.language as tl

@triton.jit
def mse_loss_fwd_kernel(
    loss_ptr,
    pred_ptr,
    target_ptr,
    batch_size,
    seq_len,
    topk,
    pred_stride_batch,
    pred_stride_seq,
    target_stride_batch,
    target_stride_seq,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    loss = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(0, topk, BLOCK_SIZE):
        k_idx = tl.arange(0, BLOCK_SIZE) + k
        mask = k_idx < topk

        pred_offset = (
            batch_idx * pred_stride_batch
            + seq_idx * pred_stride_seq
            + k_idx
        )
        target_offset = (
            batch_idx * target_stride_batch
            + seq_idx * target_stride_seq
            + k_idx
        )

        pred = tl.load(pred_ptr + pred_offset, mask=mask, other=0.0)
        target = tl.load(target_ptr + target_offset, mask=mask, other=0.0)

        diff = pred - target
        loss += tl.where(mask, diff * diff, 0.0)

    loss = tl.sum(loss) / topk
    tl.store(loss_ptr + pid, loss)

@triton.jit
def mse_loss_bwd_kernel(
    grad_pred_ptr,
    grad_loss_ptr,
    pred_ptr,
    target_ptr,
    batch_size,
    seq_len,
    topk,
    pred_stride_batch,
    pred_stride_seq,
    target_stride_batch,
    target_stride_seq,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    grad_loss = tl.load(grad_loss_ptr + pid)

    for k in range(0, topk, BLOCK_SIZE):
        k_idx = tl.arange(0, BLOCK_SIZE) + k
        mask = k_idx < topk

        pred_offset = (
            batch_idx * pred_stride_batch
            + seq_idx * pred_stride_seq
            + k_idx
        )
        target_offset = (
            batch_idx * target_stride_batch
            + seq_idx * target_stride_seq
            + k_idx
        )

        pred = tl.load(pred_ptr + pred_offset, mask=mask, other=0.0)
        target = tl.load(target_ptr + target_offset, mask=mask, other=0.0)

        diff = pred - target
        grad = 2 * diff * grad_loss / topk
        tl.store(grad_pred_ptr + pred_offset, grad, mask=mask)

class FusedMSELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        batch_size, seq_len, topk = pred.shape
        loss = torch.empty(batch_size * seq_len, device=pred.device, dtype=torch.float32)

        BLOCK_SIZE = triton.next_power_of_2(min(topk, 1024))
        grid = (batch_size * seq_len,)

        mse_loss_fwd_kernel[grid](
            loss,
            pred,
            target,
            batch_size,
            seq_len,
            topk,
            pred.stride(0),
            pred.stride(1),
            target.stride(0),
            target.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(pred, target)
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.topk = topk

        return loss.view(batch_size, seq_len)

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        batch_size, seq_len, topk = ctx.batch_size, ctx.seq_len, ctx.topk

        grad_pred = torch.empty_like(pred)
        BLOCK_SIZE = triton.next_power_of_2(min(topk, 1024))
        grid = (batch_size * seq_len,)

        mse_loss_bwd_kernel[grid](
            grad_pred,
            grad_output.flatten(),
            pred,
            target,
            batch_size,
            seq_len,
            topk,
            pred.stride(0),
            pred.stride(1),
            target.stride(0),
            target.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_pred, None

class MSELoss(torch.nn.Module):
    def forward(self, pred, target):
        return FusedMSELoss.apply(pred, target)