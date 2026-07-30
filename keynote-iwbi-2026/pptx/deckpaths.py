#!/usr/bin/env python3
"""Locate the keynote deck (keynote-iwbi-2026) so the HTML->PPTX scripts work
whether they live in the repo's pptx/ dir OR are bundled into a portable skill
that was uploaded to the Anthropic Skills library.

Resolution order:
  1. $KEYNOTE_DECK, if it points at a dir containing slides/
  2. walk up from the current working directory for keynote-iwbi-2026/slides/
  3. the repo layout relative to this file (keynote-iwbi-2026/pptx/<thisfile>)
"""
import os
import pathlib


def deck_root():
    env = os.environ.get("KEYNOTE_DECK")
    if env:
        p = pathlib.Path(env).expanduser().resolve()
        if (p / "slides").is_dir():
            return p
    cwd = pathlib.Path.cwd().resolve()
    for d in [cwd, *cwd.parents]:
        if (d / "keynote-iwbi-2026" / "slides").is_dir():
            return d / "keynote-iwbi-2026"
        if d.name == "keynote-iwbi-2026" and (d / "slides").is_dir():
            return d
    here = pathlib.Path(__file__).resolve()
    if here.parent.name == "pptx" and (here.parents[1] / "slides").is_dir():
        return here.parents[1]
    raise SystemExit(
        "Cannot locate the keynote-iwbi-2026 deck. Set $KEYNOTE_DECK to its path, "
        "or run from inside a checkout that contains keynote-iwbi-2026/slides/."
    )


DECK = deck_root()
SLIDES = DECK / "slides"
PPTX = DECK / "pptx"
LAYOUT = PPTX / "layout"
SECTIONS = PPTX / "sections"


def bg_motif():
    """bg_motif.png: prefer one bundled next to these scripts, else the repo's."""
    here = pathlib.Path(__file__).resolve().parent
    cand = here / "assets" / "bg_motif.png"
    return cand if cand.exists() else (PPTX / "assets" / "bg_motif.png")
