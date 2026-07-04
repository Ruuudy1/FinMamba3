"""LM decodability dose-response (plan-goal.md Sec. 7.3 item 1): turn lm_film.py's two-point boundary
(weakly-decodable LOB axes collapse, strongly-decodable LM regime persists) into a curve. Corrupts the
router's domain-supervision label at rate p in {0, .1, .2, .3, .4, .5} (independently per training step,
not per corpus), which caps the achievable router accuracy near 1 - p while leaving the LM's actual input
distribution (and hence the true underlying regime) untouched. Reusing lm_film.py's exact model, optimizer,
and hyperparameters isolates decodability as the only varying factor between runs.
"""
# region imports
import argparse
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
from mamba_ssm import Mamba
# endregion
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 128
BATCH = 32
D_MODEL = 128
N_LAYER = 2
STEPS = 2500
LOG_EVERY = 25
CAP_CHARS = 200000


def load_domain_text(patterns, cap):
    """Concatenate the files matching the given globs into one string capped at `cap` characters, so each
    regime is a single long character stream the sampler can draw contiguous windows from."""
    chunks = []
    total = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            text = open(path, encoding="utf-8", errors="ignore").read()
            chunks.append(text)
            total += len(text)
            if total >= cap:
                break
    return "".join(chunks)[:cap]


def build_corpus():
    """Build the two-regime corpus: English prose from the repository's Markdown against Python source code,
    two clearly different character distributions so the regime is real rather than a relabelling of one
    stream. Returns encoded tensors per domain plus the shared vocabulary size."""
    prose = load_domain_text(["*.md", "*.txt"], CAP_CHARS)
    code = load_domain_text(["src/finmamba3/**/*.py"], CAP_CHARS)
    vocabulary = sorted({character: True for character in prose + code})
    index_by_char = {character: index for index, character in enumerate(vocabulary)}
    encode = lambda text: torch.tensor([index_by_char[c] for c in text], dtype=torch.long)
    domain_streams = [encode(prose), encode(code)]
    return domain_streams, len(vocabulary)


def sample_batch(domain_streams, generator):
    """Draw a batch from a single randomly chosen domain so each sequence carries one regime label, the
    setting in which an honest per-sequence FiLM can either specialise to the regime or collapse to identity."""
    domain = int(torch.randint(0, len(domain_streams), (1,), generator=generator).item())
    stream = domain_streams[domain]
    starts = torch.randint(0, len(stream) - SEQ_LEN - 1, (BATCH,), generator=generator)
    inputs = torch.stack([stream[s:s + SEQ_LEN] for s in starts])
    targets = torch.stack([stream[s + 1:s + SEQ_LEN + 1] for s in starts])
    return inputs.to(DEVICE), targets.to(DEVICE), domain


class FiLMMambaLM(nn.Module):
    """A 2-layer char-level Mamba LM whose every block input is gated by an input-affine FiLM conditioned on a
    per-position INFERRED regime, the exact gauge-absorbable placement the paper studies. The router is
    supervised on a (possibly corrupted) domain label, the language analog of the LOB escalation's router
    supervision on the efficiency-ratio bucket."""

    def __init__(self, vocab_size, num_regimes, init_scale):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D_MODEL)
        self.block0 = Mamba(d_model=D_MODEL)
        self.norm0 = nn.LayerNorm(D_MODEL)
        self.router = nn.Linear(D_MODEL, num_regimes)
        self.regime_table = nn.Parameter(torch.zeros(num_regimes, 32))
        self.hyper = nn.Linear(32, 2 * D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.block1 = Mamba(d_model=D_MODEL)
        self.head_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)
        nn.init.zeros_(self.hyper.weight)
        nn.init.normal_(self.hyper.bias, std=init_scale)
        self.last_film_g = 0.0

    def context(self, tokens):
        stem = self.embed(tokens)
        return stem + self.block0(self.norm0(stem))

    def forward(self, tokens):
        hidden = self.context(tokens)
        router_logits = self.router(hidden)
        regime_vec = F.softmax(router_logits, dim=-1) @ self.regime_table
        film = self.hyper(regime_vec)
        gamma_raw, beta_raw = film[..., :D_MODEL], film[..., D_MODEL:]
        self.last_film_g = gamma_raw.detach().abs().mean().item()
        modulated = (1.0 + gamma_raw) * self.norm1(hidden) + beta_raw
        hidden = hidden + self.block1(modulated)
        return self.head(self.head_norm(hidden)), router_logits


def decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def fit_decay(film_g_by_step):
    """Fit film_g(t) = a exp(-t/tau) + c and report tau, the asymptote, and R^2, matching the diagnostic the
    paper applies to the LOB escalations so each dose-response point is summarised on the same footing."""
    steps = np.array(sorted(film_g_by_step), dtype=float)
    values = np.array([film_g_by_step[int(s)] for s in steps])
    popt, _ = curve_fit(decay, steps, values, p0=[0.2, 1000.0, 0.0], bounds=([0.0, 1.0, -0.1], [2.0, 1e6, 1.0]), maxfev=40000)
    a, tau, c = popt
    residual = values - decay(steps, *popt)
    r2 = 1.0 - np.sum(residual ** 2) / np.sum((values - values.mean()) ** 2)
    return float(a), float(tau), float(c), float(r2)


def run_one(corrupt_prob, domain_streams, vocab_size):
    """Train one FiLMMambaLM whose router-supervision label is corrupted (flipped to the wrong domain)
    independently at each step with probability corrupt_prob, and return its summary dict."""
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    corrupt_generator = torch.Generator().manual_seed(SEED + 1000)
    model = FiLMMambaLM(vocab_size, len(domain_streams), init_scale=0.15).to(DEVICE)
    film_modules = [model.router, model.hyper]
    film_params = [model.regime_table] + [p for module in film_modules for p in module.parameters()]
    film_ids = {id(p): True for p in film_params}
    base_params = [p for p in model.parameters() if id(p) not in film_ids]
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": 1.0e-3, "weight_decay": 1.0e-4},
        {"params": film_params, "lr": 5.0e-3, "weight_decay": 1.0e-4},
    ])
    film_g_by_step = {}
    router_acc_by_step = {}
    for step in range(STEPS):
        inputs, targets, domain = sample_batch(domain_streams, generator)
        logits, router_logits = model(inputs)
        lm_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        flip = torch.rand((), generator=corrupt_generator).item() < corrupt_prob
        supervised_domain = (1 - domain) if flip else domain
        domain_target = torch.full(inputs.shape, supervised_domain, dtype=torch.long, device=DEVICE)
        router_loss = F.cross_entropy(router_logits.reshape(-1, len(domain_streams)), domain_target.reshape(-1))
        loss = lm_loss + router_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            film_g_by_step[step] = model.last_film_g
            router_acc_by_step[step] = (router_logits.argmax(-1) == domain).float().mean().item()
    a, tau, c, r2 = fit_decay(film_g_by_step)
    start = film_g_by_step[0]
    end = film_g_by_step[max(film_g_by_step)]
    last_steps = sorted(router_acc_by_step)[-10:]
    plateau_router_acc = float(np.mean([router_acc_by_step[s] for s in last_steps]))
    plateau_film_g = float(np.mean([film_g_by_step[s] for s in last_steps]))
    result = {
        "corrupt_prob": corrupt_prob, "start": start, "end": end, "a": a, "tau": tau, "asymptote": c, "r2": r2,
        "plateau_film_g": plateau_film_g, "plateau_router_acc": plateau_router_acc,
        "film_g_by_step": film_g_by_step, "router_acc_by_step": router_acc_by_step,
    }
    print(
        f"[p={corrupt_prob:.1f}] router_acc(plateau)={plateau_router_acc:.3f} "
        f"film_g {start:.4f} -> {end:.4f} (plateau={plateau_film_g:.4f}) tau={tau:.0f} c={c:.4f} R^2={r2:.4f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="LM decodability dose-response")
    parser.add_argument("--probs", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--out", default="reports/lm_film_doseresponse.json")
    args = parser.parse_args()
    probs = [float(p) for p in args.probs.split(",")]
    domain_streams, vocab_size = build_corpus()
    print(f"corpus: prose {len(domain_streams[0])} chars, code {len(domain_streams[1])} chars, vocab {vocab_size}")
    results = [run_one(p, domain_streams, vocab_size) for p in probs]
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
