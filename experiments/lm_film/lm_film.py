"""B3 (lm-film.md): a minimal Mamba language-model control for the regime-FiLM null, to answer the venue
objection that a language-model paper has no language experiment. Our thesis is optimizer-driven and
dataset-invariant, so it predicts the same gauge-absorption decay on language; the most relevant possible
support is therefore cheap to obtain.

A 2-layer char-level Mamba LM is built directly on upstream mamba_ssm blocks over a two-domain corpus (English
prose drawn from the repo's Markdown versus Python source code), with the per-sequence domain treated as the
"regime" and fed through the same input-affine FiLM the paper studies (gamma = 1 + raw, beta = raw, applied to
each block's pre-norm input so it routes into the block's Delta/B/C projections). The modulation is forced
active at initialisation; we log film_g = mean|gamma - 1| and fit a*exp(-t/tau) + c to its trajectory.

Pre-registered (verbatim from colm-submission-goal.md B3):
  SUPPORTS: film_g decays to identity with the same signature (tau same order, R^2 high). Then add a
    "Language-model control" subsection; this upgrades the dataset-invariant / optimizer-driven claim with the
    most relevant dataset and answers the venue objection.
  CONTRADICTS: the LM FiLM persists. Then the dataset-invariance claim is falsified for language; restrict the
    paper's scope to LOB / financial sequence models and say so plainly.
"""
# region imports
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
    per-position INFERRED regime, the exact gauge-absorbable placement the paper studies. The regime is read
    from the input the host also sees (a per-step router on the stem), so the host projections can in principle
    represent the modulation themselves -- the condition under which the affine is a gauge direction -- rather
    than being handed an external domain label the host cannot see (which would be non-absorbable by
    construction). The router is supervised on the true domain, the language analog of the LOB escalation that
    supervises the router on the efficiency-ratio bucket."""

    def __init__(self, vocab_size, num_regimes, init_scale):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D_MODEL)
        self.block0 = Mamba(d_model=D_MODEL)
        self.norm0 = nn.LayerNorm(D_MODEL)
        # The router reads the contextual state after block0, not a single character embedding, so it can discriminate prose from code at all.
        self.router = nn.Linear(D_MODEL, num_regimes)
        self.regime_table = nn.Parameter(torch.zeros(num_regimes, 32))
        self.hyper = nn.Linear(32, 2 * D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.block1 = Mamba(d_model=D_MODEL)
        self.head_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)
        # Zero weight + normal bias forces the FiLM active at init (film_g ~ 0.12) on block1's input, the gauge-absorbable placement the paper's escalation studies.
        nn.init.zeros_(self.hyper.weight)
        nn.init.normal_(self.hyper.bias, std=init_scale)
        self.last_film_g = 0.0

    def context(self, tokens):
        """Return the contextual hidden state after the first block, the representation the router reads so the
        inferred regime depends on sequence context the host also sees."""
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
    paper applies to the LOB escalations so the language trajectory is summarised on the same footing."""
    steps = np.array(sorted(film_g_by_step), dtype=float)
    values = np.array([film_g_by_step[int(s)] for s in steps])
    popt, _ = curve_fit(decay, steps, values, p0=[0.2, 1000.0, 0.0],
                        bounds=([0.0, 1.0, -0.1], [2.0, 1e6, 1.0]), maxfev=40000)
    a, tau, c = popt
    residual = values - decay(steps, *popt)
    r2 = 1.0 - np.sum(residual ** 2) / np.sum((values - values.mean()) ** 2)
    return float(a), float(tau), float(c), float(r2)


def main():
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    domain_streams, vocab_size = build_corpus()
    print(f"corpus: prose {len(domain_streams[0])} chars, code {len(domain_streams[1])} chars, vocab {vocab_size}")
    model = FiLMMambaLM(vocab_size, len(domain_streams), init_scale=0.15).to(DEVICE)
    # The FiLM/router parameters get a higher LR (the LRMult analog), the same regime as the LOB escalation, for a fair comparison.
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
        domain_target = torch.full(inputs.shape, domain, dtype=torch.long, device=DEVICE)
        router_loss = F.cross_entropy(router_logits.reshape(-1, len(domain_streams)), domain_target.reshape(-1))
        loss = lm_loss + router_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            film_g_by_step[step] = model.last_film_g
            router_acc_by_step[step] = (router_logits.argmax(-1) == domain).float().mean().item()
            print(f"[loss] step={step} lm={lm_loss.item():.3f} router_acc={router_acc_by_step[step]:.2f} film_g={model.last_film_g:.4f}")
    a, tau, c, r2 = fit_decay(film_g_by_step)
    start = film_g_by_step[0]
    end = film_g_by_step[max(film_g_by_step)]
    decays_to_identity = c < 0.05 and end < 0.5 * start and r2 > 0.8
    verdict = "SUPPORTS" if decays_to_identity else "CONTRADICTS"
    result_by_key = {"start": start, "end": end, "a": a, "tau": tau, "asymptote": c, "r2": r2,
                     "verdict": verdict, "vocab": vocab_size, "steps": STEPS}
    print(f"\nfilm_g {start:.4f} -> {end:.4f};  fit a={a:.4f} tau={tau:.0f} c={c:.4f} R^2={r2:.4f}")
    print(f"Verdict: {verdict} (language-model FiLM "
          f"{'decays to identity, same signature' if verdict == 'SUPPORTS' else 'persists'})")
    json.dump(result_by_key, open("reports/lm_film.json", "w"), indent=2)
    print("wrote reports/lm_film.json")


if __name__ == "__main__":
    main()
