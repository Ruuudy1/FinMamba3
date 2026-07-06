# region imports
import torch
import torch.nn as nn
# endregion


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)
        self.mean, self.var, self.count = self.update_mean_var_count(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def update_mean_var_count(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count


class VecNormalize(nn.Module):
    def __init__(self, shape, device, epsilon=1e-8, clipob=10.0):
        super(VecNormalize, self).__init__()
        self.ob_rms = RunningMeanStd(shape, device)
        self.epsilon = epsilon
        self.clipob = clipob

    def forward(self, x):
        # Accept either (B, D) or (B, L, D); the running stats are always over the last axis.
        if x.dim() == 2:
            x_flat = x
        elif x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.view(-1, D)
        else:
            raise ValueError("Unsupported input shape")
        self.ob_rms.update(x_flat)
        mean = self.ob_rms.mean
        var = self.ob_rms.var
        x_normalized = torch.clamp((x_flat - mean) / torch.sqrt(var + self.epsilon), -self.clipob, self.clipob)
        if x.dim() == 2:
            return x_normalized
        elif x.dim() == 3:
            return x_normalized.view(B, L, D)
