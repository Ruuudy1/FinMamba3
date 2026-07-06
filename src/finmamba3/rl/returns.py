# region imports
import torch
# endregion


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise.
    inv_termination = (termination * -1) + 1
    batch_size, batch_length = rewards.shape[:2]
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device=rewards.device)
    gamma_return[:, -1] = values[:, -1]
    # Bootstrap from the last value, then fold rewards backward.
    for t in reversed(range(batch_length)):
        gamma_return[:, t] = (
            rewards[:, t] +
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] +
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
        )
    return gamma_return[:, :-1]
