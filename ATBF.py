
import torch
import torch.nn.functional as F

def ATBF(time_series, window_size=8, sigma_s_factor=0.7,
                           sigma_r_factor=0.7,block_num=10):
    time_series = torch.from_numpy(time_series)
    half_window = window_size // 2
    pad_size = half_window
    block_size = len(time_series) // block_num
    block_ts = torch.split(time_series, block_size, dim=0)
    block_ts = torch.stack(block_ts, dim=0)
    _, seq_len, num_vars = block_ts.shape

    padded_series = F.pad(block_ts.unsqueeze(0), (0, 0, pad_size, pad_size), mode='reflect').squeeze()

    spatial_weights = torch.exp(
        -0.5 * (torch.arange(-half_window, half_window + 1, dtype=torch.float32) ** 2) / (sigma_s_factor ** 2)).to(
        time_series.device)
    spatial_weights = spatial_weights.view(-1, 1).unsqueeze(0).repeat(block_num, 1, num_vars)
    filtered_series = torch.zeros_like(block_ts)

    for i in range(seq_len):
        local_start = max(0, i - half_window)
        local_end = min(seq_len, i + half_window + 1)
        local_window = padded_series[:, local_start + pad_size:local_end + pad_size, :]
        local_var = torch.var(local_window, dim=1, keepdim=True) + 1e-6
        sigma_r = sigma_r_factor * local_var.sqrt()
        intensity_weights = torch.exp(
            -0.5 * ((padded_series[:, i + pad_size, :].unsqueeze(1) - local_window) ** 2) / (sigma_r ** 2))
        combined_weights = spatial_weights[:, :local_window.size(1), :] * intensity_weights
        combined_weights = torch.softmax(combined_weights, dim=-1)
        filtered_series[:, i, :] = torch.sum(combined_weights * local_window, dim=1)

    return filtered_series
