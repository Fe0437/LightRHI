#!/usr/bin/env python3
"""Generate a shaders_registry.generated.json manifest by scanning a directory of
Slang sources for their [shader("...")] entry points.

The manifest is what cmake/Shaders.cmake reads to know which (source, entryPoint)
pairs to compile, and under which artifact name. It is a build artifact — never
edit or commit it; add a [shader(...)] entry point to a .slang file and it shows
up here on the next configure.

Artifact names are derived, not authored: <source-stem>_<entryPoint>. Consumers
compose the same name from the pair they already pass to their artifact loader,
so no name is ever written by hand on either side.
"""

import argparse
import json
import pathlib
import re
import sys

# [shader("compute")] [numthreads(8, 8, 1)] void entry_point(uint3 tid : SV_DispatchThreadID)
_ENTRY_POINT = re.compile(
    r'\[\s*shader\s*\(\s*"(?P<stage>\w+)"\s*\)\s*\]'   # the stage attribute
    r'(?:\s*\[[^\]]*\])*'                               # any further attributes ([numthreads], ...)
    r'[^{};]*?'                                         # the return type of the function it decorates
    r'\b(?P<name>\w+)\s*\(',                            # ...and the function's name
    re.DOTALL,
)

# Slang spells the pixel stage both ways; the RHI manifest only knows "fragment".
_STAGE_ALIASES = {"pixel": "fragment"}
_KNOWN_STAGES = {"vertex", "fragment", "compute"}


def scan_source(path: pathlib.Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    entries = []
    for match in _ENTRY_POINT.finditer(text):
        stage = _STAGE_ALIASES.get(match["stage"], match["stage"])
        if stage not in _KNOWN_STAGES:
            raise ValueError(f'{path.name}: unsupported shader stage "{match["stage"]}" on {match["name"]}')
        entries.append({
            "name": f"{path.stem}_{match['name']}",
            "source": path.name,
            "entryPoint": match["name"],
            "stage": stage,
        })
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shaders-dir", required=True, type=pathlib.Path,
                        help="Directory of .slang sources to scan")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="Manifest to write (shaders_registry.generated.json)")
    args = parser.parse_args()

    shaders = []
    for source in sorted(args.shaders_dir.glob("*.slang")):
        shaders.append(scan_source(source))

    manifest = {"shaders": [entry for source_entries in shaders for entry in source_entries]}
    text = json.dumps(manifest, indent=4) + "\n"

    # Only rewrite when the content actually changes: the manifest is a configure
    # -time dependency of every shader compile rule, so touching it needlessly
    # would rebuild every shader.
    if args.output.exists() and args.output.read_text(encoding="utf-8") == text:
        return 0

    args.output.write_text(text, encoding="utf-8")
    print(f"[generate_shader_registry] {args.output.name}: {len(manifest['shaders'])} entry points "
          f"from {len(shaders)} source(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
