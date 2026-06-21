#!/usr/bin/env python3
"""Compile Slang shaders listed in a shaders_registry.generated.json manifest
to backend-specific artifacts.

This is the tooling/build-layer step in the shader pipeline:

    .slang source --(slangc)--> backend-specific artifact
        Vulkan: <output-dir>/vulkan/<name>.generated.spv
        Metal:  <output-dir>/metal/<name>.generated.metal

It never emits C++ — the RHI loads these artifact files directly (see
tests/shader_artifact_loader.h). Slang is a tooling concern only; the RHI
core and backends do not know about it.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

_STAGE_TO_SLANG_STAGE = {
    "vertex": "vertex",
    "fragment": "fragment",
    "compute": "compute",
}

_TARGETS = {
    "vulkan": ("spirv", ".spv"),
    "metal": ("metal", ".metal"),
}


def load_manifest(manifest_path: pathlib.Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest_path, manifest)
    return manifest


def validate_manifest(manifest_path: pathlib.Path, manifest: dict) -> None:
    """Validate against shaders_registry.schema.json if the optional
    jsonschema package is installed; otherwise fall back to a minimal manual
    check so the manifest shape is still verified without adding a hard
    dependency. The schema lives next to this script (tools/) since it's
    generic to the whole module's shader registries, not test-specific."""
    schema_path = pathlib.Path(__file__).parent / "shaders_registry.schema.json"
    try:
        import jsonschema  # type: ignore

        schema = json.loads(schema_path.read_text())
        jsonschema.validate(instance=manifest, schema=schema)
        return
    except ImportError:
        pass

    if "shaders" not in manifest or not isinstance(manifest["shaders"], list):
        raise ValueError(f"{manifest_path}: missing top-level 'shaders' array")
    for entry in manifest["shaders"]:
        for key in ("name", "source", "entryPoint", "stage"):
            if key not in entry:
                raise ValueError(f"{manifest_path}: shader entry missing '{key}': {entry}")
        if entry["stage"] not in _STAGE_TO_SLANG_STAGE:
            raise ValueError(f"{manifest_path}: unknown stage '{entry['stage']}' in {entry}")


# ---------------------------------------------------------------------------
# Metal MSL postprocessing — TEMPORARY workaround for a Slang codegen bug.
#
# slangc 2026.13 (-target metal): when a shader uses DescriptorHandle<T>
# together with RayQuery, the emitted MSL references Slang core-module structs
# by their unsuffixed name-hint (e.g. `RayDesc`) without ever emitting the
# struct definition, so Apple's Metal compiler rejects the source with
# "unknown type name". The IR is fine — `slangc -dump-ir` shows the struct
# fully intact through the last pass, and the same shader without any
# DescriptorHandle emits a correct `struct RayDesc_0 {...}` — so the fix is
# purely textual: inject the known definition when it's referenced but absent.
#
# The definitions below are copied verbatim from what slangc itself emits in
# the working (non-DescriptorHandle) case, unsuffixed to match the name-hint
# fallback. They're thread-local-only types, so no ABI/layout concern.
#
# DELETE this whole section once the upstream bug is fixed — detection is
# self-disabling (nothing is injected when slangc emits the struct itself).
# ---------------------------------------------------------------------------

_METAL_MISSING_STRUCT_DEFS = {
    "RayDesc": (
        "struct RayDesc\n"
        "{\n"
        "    float3 Origin;\n"
        "    float TMin;\n"
        "    float3 Direction;\n"
        "    float TMax;\n"
        "};\n"
    ),
}

_METAL_INJECT_ANCHOR = "using namespace metal;"

# The push-constant buffer slot LightRHI's Metal backend always binds at
# (kPushConstantSlot in metal_command_list.cpp) — see
# _inject_argument_buffer_ids's doc comment for why this specific struct
# needs [[id(n)]] annotations.
_METAL_PUSH_CONSTANT_BUFFER_SLOT = 30


def postprocess_metal_source(text: str) -> tuple[str, list[str]]:
    """Injects definitions for core-module structs that the MSL references but
    never defines. Returns (possibly-modified text, list of injected names)."""
    injected = []
    for name, definition in _METAL_MISSING_STRUCT_DEFS.items():
        used = re.search(rf"\b{name}\b", text)
        defined = re.search(rf"\bstruct\s+{name}\b", text)
        if used and not defined:
            injected.append(name)
    if not injected:
        return text, []

    block = "\n" + "\n".join(_METAL_MISSING_STRUCT_DEFS[n] for n in injected)
    idx = text.find(_METAL_INJECT_ANCHOR)
    if idx >= 0:
        insert_at = idx + len(_METAL_INJECT_ANCHOR)
        text = text[:insert_at] + block + text[insert_at:]
    else:
        text = block + text
    return text, injected


def inject_argument_buffer_ids(text: str) -> tuple[str, bool]:
    """Metal argument buffers (a struct embedding texture/sampler/buffer
    resources, passed as a single `constant` pointer — Slang's lowering of a
    PC struct containing a DescriptorHandle<Texture2D>/<SamplerState> field)
    require every member to carry an explicit `[[id(n)]]` attribute for
    reliable codegen. Apple's own MTLArgumentEncoder docs warn: "A runtime
    validation error occurs if you create a MTLArgumentEncoder instance
    using structures that don't reference any other resources and don't
    provide any [[id()]] annotation on any of their members." Slang's Metal
    backend does not emit these annotations itself (confirmed: the raw
    generated struct has none) — without them, sampling through a bound
    texture+sampler pair was found to be non-deterministic, corrupting some
    SIMD lanes' reads unpredictably from run to run (see
    bindless_texture_test.slang's header comment for the full story).

    Finds the struct type bound at buffer(_METAL_PUSH_CONSTANT_BUFFER_SLOT)
    and, if it embeds at least one texture/sampler member, injects
    sequential [[id(0)]], [[id(1)]], ... annotations in declaration order —
    matching the order LightRHI's MetalDevice::CreateComputePipeline reads
    back via pipeline reflection (member->argumentIndex()), so the ids this
    injects and the ids the C++ side already assumes stay consistent.
    """
    m = re.search(rf"(\w+)\s+constant\s*\*\s*\w+\s*\[\[buffer\({_METAL_PUSH_CONSTANT_BUFFER_SLOT}\)\]\]", text)
    if not m:
        return text, False
    struct_name = m.group(1)

    struct_re = re.compile(rf"(struct\s+{re.escape(struct_name)}\s*\{{)(.*?)(\}}\s*;)", re.DOTALL)
    sm = struct_re.search(text)
    if not sm:
        return text, False

    body = sm.group(2)
    if "texture" not in body and "sampler" not in body:
        return text, False  # plain-data-only struct: setBytes works fine as-is

    out_lines = []
    next_id = 0
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped and stripped.endswith(";") and "[[id(" not in line:
            line = line.rstrip()[:-1] + f" [[id({next_id})]];"
            next_id += 1
        out_lines.append(line)

    text = text[:sm.start()] + sm.group(1) + "\n".join(out_lines) + sm.group(3) + text[sm.end():]
    return text, True


def compile_one(slangc: str, shaders_root: pathlib.Path, entry: dict, target: str, ext: str,
                 out_dir: pathlib.Path, include_dirs: list[pathlib.Path] | None = None,
                 defines: list[str] | None = None) -> None:
    name = entry["name"]
    source = shaders_root / entry["source"]
    entry_point = entry["entryPoint"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.generated{ext}"

    cmd = [slangc, "-target", target, "-entry", entry_point]
    if target == "spirv":
        # GPU-addressed structs are shared with C++; emit their C-compatible
        # member layout and satisfy Vulkan physical-storage-buffer validation.
        cmd += ["-fvk-use-c-layout"]
    for include_dir in include_dirs or []:
        cmd += ["-I", str(include_dir)]
    for define in defines or []:
        cmd += ["-D", define]
    cmd += ["-o", str(out_path), str(source)]
    print(f"[compile_shaders] {source.name}:{entry_point} -> {out_path}")
    subprocess.run(cmd, check=True)

    if target == "metal":
        text = out_path.read_text()
        text, injected = postprocess_metal_source(text)
        text, ids_injected = inject_argument_buffer_ids(text)
        if injected or ids_injected:
            out_path.write_text(text)
            if injected:
                print(f"[compile_shaders]   postprocessed {out_path.name}: injected missing "
                      f"struct definition(s): {', '.join(injected)} (slang metal codegen workaround)")
            if ids_injected:
                print(f"[compile_shaders]   postprocessed {out_path.name}: injected [[id(n)]] "
                      f"argument-buffer annotations (slang metal codegen workaround)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shaders-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path,
                         help="Base output dir; vulkan/ and metal/ subdirectories are created inside it")
    parser.add_argument("--slangc", required=True, help="Path to the slangc compiler")
    parser.add_argument("--backend", choices=["vulkan", "metal", "all"], default="all")
    parser.add_argument("--only", help="Compile only the manifest entry with this 'name' "
                                        "(lets the build system invoke this script once per "
                                        "shader, in parallel, instead of once for the whole manifest)")
    parser.add_argument("--include-dir", action="append", dest="include_dirs", default=[],
                         type=pathlib.Path,
                         help="Extra directory for slangc to resolve #include \"...\" against "
                              "(repeatable) — needed when a shader #includes a header living "
                              "outside its own source directory (e.g. a project's shared "
                              "CPU/GPU struct-definition headers).")
    parser.add_argument("--define", action="append", dest="defines", default=[],
                        help="Slang preprocessor definition as NAME or NAME=VALUE (repeatable).")
    args = parser.parse_args()

    manifest = load_manifest(args.shaders_json)
    shaders_root = args.shaders_json.parent

    entries = manifest["shaders"]
    if args.only is not None:
        entries = [e for e in entries if e["name"] == args.only]
        if not entries:
            raise ValueError(f"{args.shaders_json}: no shader entry named '{args.only}'")

    backends = _TARGETS.keys() if args.backend == "all" else [args.backend]
    for entry in entries:
        for backend in backends:
            target, ext = _TARGETS[backend]
            compile_one(args.slangc, shaders_root, entry, target, ext, args.output_dir / backend,
                        args.include_dirs, args.defines)

    return 0


if __name__ == "__main__":
    sys.exit(main())
