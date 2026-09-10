"""Sphinx configuration for the LightRHI documentation.

`docs/` holds documents and nothing else, so this configuration and the pinned requirements live
here in `tools/docs/` beside the script that uses them. Sphinx is pointed at this directory with
`-c`.

Nothing generated is written into `docs/`. `build_docs.py` stages the prose into `build/docs/src/`
and builds from there into `build/docs/html/`, so every produced file lives under `build/`.

The staged entry point is `index.md`, renamed from `docs/INDEX.md` while staging: the repository
names its documents in upper case, and a published site answers on `index.html`.
"""

from __future__ import annotations

project = "LightRHI"
author = "LightRHI contributors"
copyright = "2026, LightRHI contributors"

extensions = ["myst_parser"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "LightRHI"

# The documents link to sources and to the README, which live outside the staged tree. Those are
# real links for a reader on the repository, not broken document references.
suppress_warnings = ["myst.xref_missing"]

nitpicky = False
