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

Note: typeset against the COLM-2026 template as the closest available proxy; drop
the body into a venue's own style file for camera-ready if it differs. Do not link
this folder to any author-identifying repository until after the review decisions.
