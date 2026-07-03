# COLM 2026 workshop submissions

Anonymized, double-blind, non-archival. Each `.tex` is self-contained (inline
bibliography); the COLM style files (`colm2026_conference.{sty,bst}`, `natbib.sty`,
`fancyhdr.sty`) and `figures/` are bundled, so the folder builds standalone.

| File | Venue | Deadline (AoE) | Main-body limit | Main body |
|---|---|---|---|---|
| `sufm.tex` | Scientific Understanding of Foundation Models | Jun 23, 2026 | 9 pp (refs uncounted) | 9 pp |
| `actionable-interpretability.tex` | Actionable Interpretability | Jun 24, 2026 | 9 pp (refs + appendix uncounted) | 9 pp |
| `moss-paper.tex` | Methods & Opportunities at Small Scale | Jun 30, 2026 | 4 pp (unlimited supplementary) | 4 pp |

Build (run twice for refs):

    pdflatex -interaction=nonstopmode <paper>.tex
    pdflatex -interaction=nonstopmode <paper>.tex

One core result (input-affine FiLM conditioning of a Mamba selective scan is
gauge-absorbable and decays to identity under joint training), three venue-specific
framings: understanding (SUFM), practitioner prescriptions (Actionable), and a
small-scale methods package (MOSS).

Template status (confirmed against each CFP, 2026-06-22):
- `sufm.tex` — SUFM mandates the **default COLM template** (science-ai-2026.github.io);
  the current typesetting is correct as-is. Double-blind, OpenReview group `.../Workshop/SUFM`.
- `actionable-interpretability.tex` — Actionable Interpretability mandates the **COLM
  template** (actionable-interpretability.github.io/cfp); correct as-is. Double-blind,
  OpenReview group `.../Workshop/AIW`.
- `moss-paper.tex` — MOSS distributes **its own style files** (not COLM); the body must be
  dropped into the MOSS template and the 4-page body re-verified there before submission.
  CFP: sites.google.com/view/moss-colm-2026/call-for-papers. Double-blind, OpenReview group
  `colmweb.org/COLM/2026/Workshop/MOSS`.

Do not link this folder to any author-identifying repository until after the review decisions.
