# MOSS @ COLM 2026 — submission build (official MOSS template)

This is `moss-paper.tex` re-typeset on the **MOSS workshop's own style files**
(ICML-2025 one-column template), as the MOSS CFP requires its template rather than
the COLM one. Body text is identical to the COLM-proxy `../moss-paper.tex`; only the
template wrapper (preamble + title/author block) differs.

- **Blind review (submit this):** `\usepackage{icml2025}` — auto-anonymizes the author
  block to "Anonymous Authors", prints the "under review" notice, and adds line numbers.
- **Camera-ready (if accepted):** swap to `\usepackage[accepted]{MOSS_camera_ready}` and
  fill in the real `\icmlauthor`/`\icmlaffiliation` entries.

Build (run twice for refs + line numbers):

    pdflatex -interaction=nonstopmode moss-paper.tex
    pdflatex -interaction=nonstopmode moss-paper.tex

Verified: 8 pp total, **main body §1–§7 within 4 pages** (appendix starts on p.5),
0 errors / 0 undefined refs / 0 overfull hboxes; page 1 anonymized.
(2026-06-30 clarity pass: split dense sentences, added a non-finance LOB primer,
and glossed Brier/PnL, which fills page 4 fully rather than leaving headroom, but
the main body still ends exactly at the 4-page boundary. 2026-07-02 review fixes
grew the appendix to 8 pp total; 2026-07-04 edits, tau CIs/dt_bias/decodability
sentence/nits, held the body at exactly 4 pp again after a compression pass.)

CFP: https://sites.google.com/view/moss-colm-2026/call-for-papers ·
OpenReview group `colmweb.org/COLM/2026/Workshop/MOSS` · 4 pp body + unlimited
supplementary (refs + appendix uncounted) · double-blind · non-archival · Jun 30 AoE.

Notes: the blind footer is overridden to a MOSS-specific notice (preamble
`\renewcommand{\Notice@String}{...}`). MOSS additionally allows +1 body page for
the camera-ready (final) version, so there is a safety margin if any future edit
pushes past 4 pages in the blind build.
