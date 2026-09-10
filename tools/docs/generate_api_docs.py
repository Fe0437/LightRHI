#!/usr/bin/env python3
"""Turn the brace-tag comments in the project's public C/C++ API into MyST pages for Sphinx.

It reads the `{brief}` / `{param}` brace tags the project writes in its C and C++ comments, recovers
the declaration each one documents, and writes one MyST page per module or header using the Sphinx C
and C++ domains. Each page also carries a PlantUML class diagram of the module, written inline as a `{uml}` block:
Sphinx renders it during the build, so there is no separate diagram step and no rendered image to
keep anywhere. A diagram longer than `MAXIMUM_INLINE_DIAGRAM` lines is written beside the page
instead, so a page stays readable in source form.

Scope is the public API: the C++20 modules under `src/`, the frozen ABI headers under
`src/host/abi/public/`, and the optional C++ client. Tests, examples and private implementation
files carry documentation too, but they are not the surface a consumer reads.

The inline tags — `{apiref}`, `{code}`, `{param}`, `{see}` — are already valid MyST role syntax, so
they pass through untouched.


Pages are written to the directory `--output` names, which is a staging directory under `build/`.
Standard library only. Idempotent: running it twice produces byte-identical output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil

# The C/C++ lexer this pipeline depends on: it splits a translation unit into code and comment
# spans, understanding string, character and raw-string literals so a `/` or a quote inside one is
# never mistaken for the start of a comment.


def _regions(source: str) -> list[tuple[str, int, int]]:
    """Split source into ('code' | 'block' | 'line', start, end) spans covering the whole text.

    The lexer only needs to be right about where comments begin and end, which means it has to
    recognise every construct a `/` or a quote can hide inside: string and character literals, raw
    strings with their custom delimiters, and escape sequences.
    """
    spans: list[tuple[str, int, int]] = []
    index = 0
    length = len(source)
    code_start = 0

    while index < length:
        character = source[index]

        # A raw string swallows everything up to its closing delimiter, including quotes and slashes.
        if character in "rR" and source.startswith('"', index + 1) and _is_raw_string_prefix(source, index):
            closing = source.find("(", index)
            if closing != -1:
                delimiter = source[source.index('"', index) + 1 : closing]
                terminator = f'){delimiter}"'
                end = source.find(terminator, closing)
                index = length if end == -1 else end + len(terminator)
                continue

        if character == '"' or character == "'":
            index = _skip_quoted(source, index, character)
            continue

        if character == "/" and index + 1 < length:
            following = source[index + 1]
            if following == "/":
                end = source.find("\n", index)
                end = length if end == -1 else end
                spans.append(("code", code_start, index))
                spans.append(("line", index, end))
                code_start = end
                index = end
                continue
            if following == "*":
                end = source.find("*/", index + 2)
                end = length if end == -1 else end + 2
                spans.append(("code", code_start, index))
                spans.append(("block", index, end))
                code_start = end
                index = end
                continue

        index += 1

    spans.append(("code", code_start, length))
    return [span for span in spans if span[1] < span[2]]


def _is_raw_string_prefix(source: str, index: int) -> bool:
    """Whether the R at `index` opens a raw string rather than ending an identifier."""
    if index > 0 and (source[index - 1].isalnum() or source[index - 1] == "_"):
        return False
    return source[index] == "R"


def _skip_quoted(source: str, index: int, quote: str) -> int:
    """The index just past a string or character literal that starts at `index`."""
    cursor = index + 1
    while cursor < len(source):
        if source[cursor] == "\\":
            cursor += 2
            continue
        if source[cursor] == quote:
            return cursor + 1
        if source[cursor] == "\n" and quote == "'":
            # An unterminated character literal is not one; do not run to end of file over it.
            return cursor
        cursor += 1
    return len(source)


# The public surface, in the order the reference presents it.
# Everything project-specific lives here, and a sibling `api_docs.json` overrides any key. That file
# is what lets the same generator serve this repository and the libraries it is consumed with: the
# script stays identical in each of them, and only the configuration differs.
CONFIG: dict = {
    # (directory, glob) pairs naming the public API surface, in the order they should be read.
    "sources": [
        ["src/host/abi/public", "*.h"],
        ["apps/cxx", "*.hpp"],
        ["src", "*.cppm"],
    ],
    # A package key mapped to the heading its page carries. A key that is absent is title-cased.
    "package_titles": {
        "core": "Core",
        "components": "Components",
        "graph": "Graph",
        "canvas": "Canvas",
        "brush": "Brush",
        "use_cases": "Use cases",
        "abi": "C ABI",
        "cxx": "C++ client",
    },
    "repository_url": "https://github.com/flexible-drawing/flexible_drawing/blob/main",
    # A short note every package page opens with. Empty for a project with no glossary to point at.
    "package_note": ["Unfamiliar with a drawing term? See the {doc}`../GLOSSARY`."],
    # The prose the generated index opens with, before its table of contents.
    "intro": [
        "New to the project? Use the {doc}`../GLOSSARY` for drawing terms used in the",
        "declarations below. The {doc}`../DRAWING_GUIDE` explains how the main parts",
        "work together.",
        "",
        "Each page covers one package. It first explains the package, then lists its",
        "public types, and finally documents each declaration.",
        "",
        "This reference is generated from comments beside the C and C++ declarations.",
    ],
}

_OVERRIDES = Path(__file__).resolve().parent / "api_docs.json"
if _OVERRIDES.is_file():
    CONFIG.update(json.loads(_OVERRIDES.read_text(encoding="utf-8")))

API_SOURCES: tuple[tuple[str, str], ...] = tuple((entry[0], entry[1]) for entry in CONFIG["sources"])

GENERATED_NOTICE = "<!-- Generated by tools/docs/generate_api_docs.py. Do not edit by hand. -->"

# A diagram short enough to read in place is written into the page; only a longer one earns a file
# of its own. The same rule governs the hand-written diagrams in the guides.
MAXIMUM_INLINE_DIAGRAM = 100


@dataclass
class DocComment:
    """The tags recovered from one documentation comment."""

    brief: str = ""
    details: list[str] = field(default_factory=list)
    params: list[tuple[str, str]] = field(default_factory=list)
    tparams: list[tuple[str, str]] = field(default_factory=list)
    returns: str = ""
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    page: tuple[str, str] | None = None


@dataclass
class Entity:
    """One documented declaration and the comment attached to it."""

    kind: str
    name: str
    signature: str
    namespace: str
    doc: DocComment
    line: int
    owner: str = ""
    access: str = "public"
    members: list[tuple[str, str]] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    fields: list[tuple[str, str, str, str]] = field(default_factory=list)


@dataclass
class FileDoc:
    """Everything one translation unit contributes to the reference."""

    path: Path
    slug: str
    title: str
    module: str
    is_c_api: bool
    doc: DocComment
    entities: list[Entity]


# Ordered so `*/` is recognised before the `*` that opens an ordinary continuation line.
DECORATION = re.compile(r"^\s*(?:/\*\*<?|/\*!|///<|///|\*/|\*)[ \t]?")

TAG_LINE = re.compile(r"^\{(file|brief|returns|note|warning|pre|post)\}\s*(.*)$")
TAG_NAMED = re.compile(r"^\{(param|tparam|page)\s+([^}]+)\}\s*(.*)$")

# The same tags in Doxygen spelling. Libraries this project consumes write `\brief` rather than
# `{brief}`, and one generator serves both: a Doxygen tag is rewritten into the brace form before
# anything else looks at the line, so the rest of the parser only ever sees one convention.
DOXYGEN_LINE = re.compile(r"^[\\@](file|brief|return|returns|note|warning|pre|post)\b[ \t]*(.*)$")
DOXYGEN_NAMED = re.compile(r"^[\\@](param|tparam)\b[ \t]*(\S+)[ \t]*(.*)$")


def _normalized_tag(line: str) -> str:
    """A Doxygen tag rewritten into the brace form, or the line unchanged."""
    named = DOXYGEN_NAMED.match(line)
    if named:
        return f"{{{named.group(1)} {named.group(2)}}} {named.group(3)}".rstrip()
    simple = DOXYGEN_LINE.match(line)
    if simple:
        name = "returns" if simple.group(1) in {"return", "returns"} else simple.group(1)
        return f"{{{name}}} {simple.group(2)}".rstrip()
    return line


def _comment_content(text: str) -> list[str]:
    """The lines of a comment with `/**`, `*` and `*/` decoration removed."""
    lines: list[str] = []
    for raw in text.split("\n"):
        # The decoration is stripped exactly, never with `strip()`: indentation inside a fenced
        # example is part of the example, and a closing `*/` must not survive as a stray slash.
        match = DECORATION.match(raw)
        line = raw[match.end() :] if match else raw.lstrip()
        if line.rstrip().endswith("*/"):
            line = line.rstrip()[:-2].rstrip()
        lines.append(line)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def parse_doc(text: str) -> DocComment:
    """Read a documentation comment into its tags, keeping fenced code blocks intact."""
    doc = DocComment()
    current: list[str] | None = None
    in_fence = False

    for line in _comment_content(text):
        stripped = _normalized_tag(line.strip())

        if stripped.startswith("```"):
            in_fence = not in_fence
            (current if current is not None else doc.details).append(line)
            continue
        if in_fence:
            (current if current is not None else doc.details).append(line)
            continue

        named = TAG_NAMED.match(stripped)
        if named:
            tag, name, rest = named.group(1), named.group(2).strip(), named.group(3)
            if tag == "param":
                doc.params.append((name, rest))
                current = None
            elif tag == "tparam":
                doc.tparams.append((name, rest))
                current = None
            else:
                doc.page = (name, rest)
                current = doc.details
            continue

        simple = TAG_LINE.match(stripped)
        if simple:
            tag, rest = simple.group(1), simple.group(2)
            if tag == "file":
                current = None
            elif tag == "brief":
                doc.brief = rest
                current = None
            elif tag == "returns":
                doc.returns = rest
                current = None
            else:
                bucket = {
                    "note": doc.notes,
                    "warning": doc.warnings,
                    "pre": doc.preconditions,
                    "post": doc.postconditions,
                }[tag]
                bucket.append(rest)
                current = None
            continue

        # A continuation line extends whatever tag was last opened.
        if current is not None:
            current.append(line)
        elif doc.params and stripped and not doc.details:
            name, existing = doc.params[-1]
            doc.params[-1] = (name, f"{existing} {stripped}".strip())
        elif doc.notes and stripped and not doc.details:
            doc.notes[-1] = f"{doc.notes[-1]} {stripped}".strip()
        else:
            doc.details.append(line)

    while doc.details and not doc.details[0].strip():
        doc.details.pop(0)
    return doc


def _body_span(blanked: str, offset: int) -> tuple[int, int] | None:
    """The bounds of the `{ ... }` body that follows a declaration.

    Braces are counted in the blanked text, where a brace inside a comment or a string cannot throw
    the depth off. The same bounds then address the original source, which is how a trailing
    comment can be read back out of a body whose structure was found safely.
    """
    opening = blanked.find("{", offset)
    if opening == -1:
        return None
    depth = 0
    for index in range(opening, len(blanked)):
        if blanked[index] == "{":
            depth += 1
        elif blanked[index] == "}":
            depth -= 1
            if depth == 0:
                return (opening + 1, index)
    return None


def _body_after(blanked: str, offset: int) -> str:
    """The `{ ... }` body that follows a declaration, with comments already blanked out."""
    span = _body_span(blanked, offset)
    return blanked[span[0] : span[1]] if span else ""


# A description written after the member it documents: `int Count{}; ///< How many.` Doxygen spells
# it `///<`, `//<` or `/**< ... */`, and all three mean the same thing here.
TRAILING_DOC = re.compile(r";[ \t]*(?:///<|//<|/\*\*<|/\*!<)[ \t]*(?P<text>.*?)[ \t]*(?:\*/)?[ \t]*$")
NAME_BEFORE_SEMICOLON = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\}|=[^;]*)?\s*;")


def _field_docs(body: str) -> dict[str, str]:
    """Member name to the description written after it on the same line.

    Read from the original text rather than the blanked one, because the description *is* the
    comment. Only the first description for a name is kept, so a nested type reusing a member name
    cannot overwrite the outer one.
    """
    docs: dict[str, str] = {}
    for line in body.split("\n"):
        trailing = TRAILING_DOC.search(line)
        if not trailing:
            continue
        name = NAME_BEFORE_SEMICOLON.search(line[: trailing.start() + 1])
        if name and trailing.group("text"):
            docs.setdefault(name.group(1), trailing.group("text"))
    return docs


def _without_nested_blocks(body: str) -> str:
    """The type's own declarations, with every method body and initialiser removed.

    A method body contains semicolons of its own, so splitting the class on `;` without this cuts
    declarations in half and leaves fragments like `} private: std::vector<...>` behind.
    """
    kept: list[str] = []
    depth = 0
    for character in body:
        if character == "{":
            depth += 1
            continue
        if character == "}":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            kept.append(character)
    return "".join(kept)


ACCESS = re.compile(r"\b(public|private|protected)\s*:")
FIELD = re.compile(r"^(?P<type>.+?[\s*&])(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])?$")
FUNCTION_POINTER = re.compile(r"^(?P<type>.+?)\(\s*\*\s*(?P<name>[A-Za-z_]\w*)\s*\)\s*\(")
SKIP_STATEMENT = re.compile(r"^\s*(?:using|typedef|static_assert|friend|template|class|struct|enum|union)\b")


def _parse_fields(body: str, default_access: str, docs: dict[str, str] | None = None) -> list[tuple]:
    """The data members a type declares, as (visibility, type, name, declaration, description).

    A class diagram without members is a box with a name on it, which the header already told you.
    The fields are the part that says what a type actually holds.
    """
    fields: list[tuple] = []
    docs = docs or {}
    access = default_access
    for raw in _without_nested_blocks(body).split(";"):
        statement = re.sub(r"\s+", " ", raw).strip()
        if not statement:
            continue

        # The last specifier in the chunk is the one the declaration after it lives under.
        specifiers = ACCESS.findall(statement)
        if specifiers:
            access = specifiers[-1]
            statement = ACCESS.sub("", statement).strip()
            if not statement:
                continue

        if SKIP_STATEMENT.match(statement):
            continue

        pointer = FUNCTION_POINTER.match(statement)
        if pointer:
            name = pointer.group("name")
            fields.append((access, pointer.group("type").strip() + "(*)()", name, statement, docs.get(name, "")))
            continue

        # A member function is documented as its own entity; only data lands here.
        if "(" in statement or ")" in statement or "{" in statement:
            continue

        match = FIELD.match(ATTRIBUTE.sub("", statement))
        if match:
            name = match.group("name")
            fields.append((access, match.group("type").strip(), name, statement, docs.get(name, "")))
    return fields


BASE_CLAUSE = re.compile(r":\s*(?P<bases>[^{]+)$")
BASE_NAME = re.compile(r"(?:public|private|protected|virtual)?\s*([A-Za-z_][\w:]*)")


def _parse_bases(signature: str) -> list[str]:
    """The types a class derives from, by their last name component."""
    without_template = re.sub(r"^template\s*<[^>]*>\s*", "", signature)
    clause = BASE_CLAUSE.search(without_template)
    if not clause:
        return []
    bases: list[str] = []
    for part in clause.group("bases").split(","):
        match = BASE_NAME.search(part.strip())
        if match:
            bases.append(match.group(1).split("::")[-1])
    return bases


def _blank_comments(source: str) -> str:
    """The source with every comment replaced by spaces, so offsets still line up."""
    pieces: list[str] = []
    for kind, start, end in _regions(source):
        text = source[start:end]
        pieces.append(text if kind == "code" else re.sub(r"[^\n]", " ", text))
    return "".join(pieces)


# A scope opener: the keyword, the name, and whatever base-clause sits between it and the brace.
SCOPE = re.compile(r"\b(?:export\s+)?(namespace|class|struct)\s+([A-Za-z_][\w:]*)[^;{]*\{")


def _scopes_at(blanked: str, offset: int) -> tuple[str, str]:
    """The namespace and the innermost enclosing type at `offset`.

    Both come from one walk over the braces. The enclosing type is what lets a member be documented
    inside the class it belongs to, rather than beside it where its name would collide with the
    class's own.
    """
    opens: dict[int, tuple[str, str]] = {}
    for match in SCOPE.finditer(blanked):
        if match.end() > offset:
            break
        opens[match.end() - 1] = (match.group(1), match.group(2))

    stack: list[tuple[int, str, str]] = []
    depth = 0
    for index, character in enumerate(blanked[:offset]):
        if character == "{":
            depth += 1
            if index in opens:
                keyword, name = opens[index]
                stack.append((depth, keyword, name))
        elif character == "}":
            while stack and stack[-1][0] >= depth:
                stack.pop()
            depth -= 1

    namespace = "::".join(name for _, keyword, name in stack if keyword == "namespace")
    types = [name for _, keyword, name in stack if keyword != "namespace"]
    return namespace, (types[-1] if types else "")


def _member_access_at(blanked: str, offset: int) -> str:
    """Access of a declaration inside its nearest class or struct."""
    opens = {match.end() - 1: match.group(1) for match in SCOPE.finditer(blanked) if match.end() <= offset}
    stack: list[tuple[int, str, str]] = []
    depth = 0
    index = 0
    while index < offset:
        character = blanked[index]
        if stack and depth == stack[-1][0]:
            access = re.match(r"(public|private|protected)\s*:", blanked[index:])
            if access:
                owner_depth, keyword, _ = stack[-1]
                stack[-1] = (owner_depth, keyword, access.group(1))
                index += access.end()
                continue
        if character == "{":
            depth += 1
            if index in opens:
                keyword = opens[index]
                default = "private" if keyword == "class" else "public"
                stack.append((depth, keyword, default))
        elif character == "}":
            while stack and stack[-1][0] >= depth:
                stack.pop()
            depth -= 1
        index += 1
    for _, keyword, access in reversed(stack):
        if keyword != "namespace":
            return access
    return "public"


ATTRIBUTE = re.compile(r"\[\[[^\]]*\]\]\s*")
DECL_NOISE = re.compile(r"^\s*(?:export|inline|static|friend)\s+")


def _declaration_after(blanked: str, source: str, offset: int) -> tuple[str, int] | None:
    """The declaration text that follows a comment, ending at its `{`, `;` or `=`."""
    index = offset
    while index < len(blanked) and blanked[index] in " \t\n\r":
        index += 1
    if index >= len(blanked):
        return None

    depth = 0
    end = index
    while end < len(blanked):
        character = blanked[end]
        if character in "([":
            depth += 1
        elif character in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and character in "{;":
            break
        elif depth == 0 and character == "=" and blanked[end : end + 2] != "==":
            # An alias is its right-hand side; stopping here would leave a bare `using Name`.
            if re.match(r"^\s*(?:export\s+)?using\b", blanked[index:end]):
                end += 1
                continue
            break
        end += 1

    text = blanked[index:end]
    if not text.strip():
        return None
    return re.sub(r"\s+", " ", text.strip()), source.count("\n", 0, index) + 1


KIND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("enum", re.compile(r"\benum\s+class\s+([A-Za-z_]\w*)")),
    ("enum", re.compile(r"\benum\s+([A-Za-z_]\w*)")),
    ("struct", re.compile(r"\bstruct\s+([A-Za-z_]\w*)")),
    ("class", re.compile(r"\bclass\s+([A-Za-z_]\w*)")),
    ("macro", re.compile(r"^#\s*define\s+([A-Za-z_]\w*)")),
    ("type", re.compile(r"^typedef\b.*?\b([A-Za-z_]\w*)$")),
    ("type", re.compile(r"^using\s+([A-Za-z_]\w*)")),
    ("var", re.compile(r"\b(?:constexpr|const)\b.*?\b([A-Za-z_]\w*)\s*$")),
)

FUNCTION = re.compile(r"([A-Za-z_]\w*)\s*\(")

# An operator's name is not an identifier, so the plain function pattern cannot see it and the
# declaration falls through to the variable rule. Every such declaration then becomes a variable
# with the same trailing word for a name, and Sphinx rejects the second one as a duplicate.
OPERATOR = re.compile(r"\boperator\s*(\(\)|\[\]|[^\s\w(]{1,3}|[A-Za-z_]\w*)\s*\(")


KEYWORDS = frozenset({"struct", "class", "enum", "union", "const", "void", "typedef"})

TYPEDEF_TAGGED = re.compile(r"^typedef\s+(struct|enum|union)\s+([A-Za-z_]\w*)")
TYPEDEF_POINTER = re.compile(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)")
TRAILING_NAME = re.compile(r"\b([A-Za-z_]\w*)\s*$")
VARIABLE = re.compile(r"\b(?:constexpr|const|inline)\b.*?\b([A-Za-z_]\w*)\s*$")


def _classify(signature: str) -> tuple[str, str] | None:
    """The kind and name a declaration introduces, or None when it introduces nothing documented."""
    cleaned = ATTRIBUTE.sub("", signature).strip()
    if cleaned.startswith(("#include", "#pragma", "module", "import", "namespace", "export namespace")):
        return None
    if cleaned.startswith("#define"):
        match = re.match(r"^#\s*define\s+([A-Za-z_]\w*)", cleaned)
        return ("macro", match.group(1)) if match else None

    if cleaned.startswith("typedef"):
        tagged = TYPEDEF_TAGGED.match(cleaned)
        if tagged:
            return ("enum" if tagged.group(1) == "enum" else "struct"), tagged.group(2)
        pointer = TYPEDEF_POINTER.search(cleaned)
        if pointer:
            return "type", pointer.group(1)
        trailing = TRAILING_NAME.search(cleaned)
        if trailing and trailing.group(1) not in KEYWORDS:
            return "type", trailing.group(1)
        return None

    # A function-pointer field inside an interface table is a member of it, not a free function.
    if TYPEDEF_POINTER.search(cleaned):
        return "member", TYPEDEF_POINTER.search(cleaned).group(1)

    # A type-introducing keyword only counts in the head, before any parameter list: a function
    # returning `struct X` is a function, and a member function ending `const noexcept` is not a
    # variable just because it says `const`.
    paren = cleaned.find("(")
    head = cleaned if paren == -1 else cleaned[:paren]
    for kind, pattern in KIND_PATTERNS:
        if kind in {"macro", "var"}:
            continue
        match = pattern.search(head)
        if match:
            return kind, match.group(1)

    if paren != -1:
        operator = OPERATOR.search(cleaned)
        if operator:
            symbol = operator.group(1)
            separator = " " if symbol[0].isalpha() or symbol[0] == "_" else ""
            return "function", f"operator{separator}{symbol}"
        candidates = FUNCTION.findall(cleaned)
        if candidates:
            return "function", candidates[0]

    variable = VARIABLE.search(cleaned)
    if variable and variable.group(1) not in KEYWORDS:
        return "var", variable.group(1)
    return None


ENUMERATOR = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:=[^,]*)?,?\s*(?://<\s*(.*?)\s*)?$")


def _enumerators(source: str, start: int) -> list[tuple[str, str]]:
    """The enumerators of an enum body, with their trailing `///<` descriptions."""
    opening = source.find("{", start)
    if opening == -1:
        return []
    closing = source.find("}", opening)
    if closing == -1:
        return []
    members: list[tuple[str, str]] = []
    for line in source[opening + 1 : closing].split("\n"):
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        match = ENUMERATOR.match(text.replace("///<", "//<"))
        if match:
            members.append((match.group(1), (match.group(2) or "").strip()))
    return members


# A `///<` comment documents the declaration on its own line and is read by the field parser, so it
# never starts a documentation comment of its own.
LINE_DOC = re.compile(r"^///(?!<)")
BLOCK_DOC_PREFIXES = ("/**", "/*!")
FILE_TAG = re.compile(r"[\\@]file\b")


def _doc_spans(source: str, regions: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
    """Spans of documentation comments, in source order.

    Two forms are recognised: a `/** ... */` block, and a run of adjacent `///` lines, which is the
    usual Doxygen spelling. A run is joined into one span so a multi-line description, and the tags
    under it, are read as a single comment rather than one comment per line.
    """
    spans: list[tuple[int, int]] = []
    index = 0
    count = len(regions)
    while index < count:
        kind, start, end = regions[index]
        if kind == "block" and source[start:end].lstrip().startswith(BLOCK_DOC_PREFIXES):
            spans.append((start, end))
            index += 1
            continue
        if kind == "line" and LINE_DOC.match(source[start:end].lstrip()):
            run_end = end
            index += 1
            # Consecutive `///` lines are separated only by the newline and indentation between them.
            while index + 1 < count:
                gap_kind, gap_start, gap_end = regions[index]
                next_kind, next_start, next_end = regions[index + 1]
                joined = (
                    gap_kind == "code"
                    and not source[gap_start:gap_end].strip()
                    and next_kind == "line"
                    and LINE_DOC.match(source[next_start:next_end].lstrip())
                )
                if not joined:
                    break
                run_end = next_end
                index += 2
            spans.append((start, run_end))
            continue
        index += 1
    return spans


def parse_file(path: Path, root: Path) -> FileDoc | None:
    """Read one translation unit into the entities its comments document."""
    source = path.read_text(encoding="utf-8")
    blanked = _blank_comments(source)

    # The trailing group captures a module partition, as in `lightphi.input:samples`. A partition
    # belongs to the package its primary module names, which `_package_key` resolves.
    module_match = re.search(r"^export\s+module\s+([\w.]+(?::[\w.]+)?)\s*;", source, re.MULTILINE)
    module = module_match.group(1) if module_match else ""

    file_doc = DocComment()
    entities: list[Entity] = []

    for start, end in _doc_spans(source, _regions(source)):
        text = source[start:end]
        if "{file}" in text or FILE_TAG.search(text):
            file_doc = parse_doc(text)
            continue

        declaration = _declaration_after(blanked, source, end)
        if declaration is None:
            continue
        signature, line = declaration
        classified = _classify(signature)
        if classified is None:
            continue
        entity_kind, name = classified

        namespace, owner = _scopes_at(blanked, start)
        entity = Entity(
            kind=entity_kind,
            name=name,
            signature=ATTRIBUTE.sub("", signature).strip(),
            namespace=namespace,
            doc=parse_doc(text),
            line=line,
            owner=owner,
            access=_member_access_at(blanked, start),
        )
        if entity_kind == "enum":
            entity.members = _enumerators(source, end)
        elif entity_kind in {"class", "struct"}:
            entity.bases = _parse_bases(entity.signature)
            span = _body_span(blanked, end)
            entity.fields = _parse_fields(
                blanked[span[0] : span[1]] if span else "",
                "private" if entity_kind == "class" else "public",
                _field_docs(source[span[0] : span[1]]) if span else {},
            )
        if entity.access == "public":
            entities.append(entity)

    relative = path.relative_to(root)
    slug = str(relative.with_suffix("")).replace("/", ".")
    return FileDoc(
        path=relative,
        slug=slug,
        title=module or relative.name,
        module=module,
        is_c_api=path.suffix == ".h",
        doc=file_doc,
        entities=entities,
    )


DIRECTIVES = {
    ("cpp", "class"): "cpp:class",
    ("cpp", "struct"): "cpp:struct",
    ("cpp", "enum"): "cpp:enum-class",
    ("cpp", "function"): "cpp:function",
    ("cpp", "type"): "cpp:type",
    ("cpp", "var"): "cpp:var",
    ("cpp", "macro"): "c:macro",
    ("c", "class"): "c:struct",
    ("c", "struct"): "c:struct",
    ("c", "enum"): "c:enum",
    ("c", "function"): "c:function",
    ("c", "type"): "c:type",
    ("c", "var"): "c:var",
    ("c", "macro"): "c:macro",
    ("c", "member"): "c:member",
    ("cpp", "member"): "cpp:member",
}


def _strip_initializer(signature: str) -> str:
    """A constructor's member-initialiser list, which is not part of the declaration."""
    depth = 0
    for index, character in enumerate(signature):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == ":" and depth == 0:
            if signature[index : index + 2] == "::" or (index and signature[index - 1] == ":"):
                continue
            return signature[:index].strip()
    return signature


def _signature_for(entity: Entity, domain: str) -> str:
    """The declaration as the Sphinx domain wants to read it."""
    signature = DECL_NOISE.sub("", entity.signature).strip()
    # The domains take the declared type, never the keyword that names it: `cpp:type` reads
    # `Name = type`, and `c:type` reads the typedef-like declaration without its `typedef`.
    signature = re.sub(r"^(?:typedef|using)\s+", "", signature)

    if entity.kind == "macro":
        definition = re.sub(r"^#\s*define\s+", "", signature)
        match = re.match(r"^([A-Za-z_]\w*)(\([^)]*\))?", definition)
        return match.group(0) if match else definition.split(" ")[0]
    if entity.kind == "function":
        return _strip_initializer(signature)
    if entity.kind in {"class", "struct", "enum"}:
        if domain == "c":
            return entity.name
        template = re.match(r"^template\s*<[^>]*>", signature)
        prefix = f"{template.group(0)} " if template else ""
        body = signature[len(prefix) :] if prefix else signature
        # The directive already says which kind this is; the keyword would be read as part of a name.
        body = re.sub(r"^(?:enum\s+class|enum|class|struct|union)\s+", "", body)
        return (prefix + body).strip()
    return signature


def _render_doc(doc: DocComment, indent: str = "") -> list[str]:
    """The body of one entity, as MyST."""
    lines: list[str] = []
    if doc.brief:
        lines.append(f"{indent}{doc.brief}")
        lines.append("")
    for detail in doc.details:
        lines.append(f"{indent}{detail}" if detail.strip() else "")
    if doc.details:
        lines.append("")
    for name, description in doc.tparams:
        lines.append(f"{indent}- **`{name}`** (template) — {description}")
    for name, description in doc.params:
        lines.append(f"{indent}- **`{name}`** — {description}")
    if doc.params or doc.tparams:
        lines.append("")
    if doc.returns:
        lines.append(f"{indent}**Returns** — {doc.returns}")
        lines.append("")
    for entries, style in (
        (doc.preconditions, "note"),
        (doc.notes, "note"),
        (doc.warnings, "warning"),
        (doc.postconditions, "note"),
    ):
        for entry in entries:
            lines.append(f"{indent}```{{{style}}}")
            lines.append(f"{indent}{entry}")
            lines.append(f"{indent}```")
            lines.append("")
    return lines


DIAGRAM_BOX = re.compile(r"^(?:class|enum)\s+([A-Za-z_]\w*)", re.M)


def _diagram_legend(diagram: str, index: TypeIndex) -> list[str]:
    """Links to every type the diagram draws.

    A rendered diagram is a picture: the boxes cannot be clicked. This puts each one back within
    reach, including the types drawn from other modules, which is where a reader usually wants to go
    next.
    """
    names = [name for name in DIAGRAM_BOX.findall(diagram) if name in index.entities]
    if not names:
        return []
    references: list[str] = []
    for name in names:
        entity = index.entities[name]
        document = index.owners[name]
        role = C_ROLES[entity.kind] if document.is_c_api else "cpp:any"
        scope = "::".join(part for part in (entity.namespace, entity.owner) if part)
        qualified = f"{scope}::{name}" if scope else name
        references.append(f"{{{role}}}`{name} <{qualified}>`")
    return ["In the diagram: " + " · ".join(references), ""]


def render_module_section(
    document: FileDoc, repository_url: str, diagram: str | None, index: TypeIndex, level: int = 2
) -> list[str]:
    """One module, as a section of the page its package owns."""
    domain = "c" if document.is_c_api else "cpp"
    heading = "#" * level
    lines = [f"{heading} {document.title}", ""]

    if document.doc.brief:
        lines += [f"{document.doc.brief}", ""]
    if document.doc.details:
        lines += [*document.doc.details, ""]

    lines += ["```{admonition} Source", ":class: seealso", f"[`{document.path}`]({repository_url}/{document.path})"]
    if document.module:
        lines.append(f"— module `{document.module}`")
    lines += ["```", ""]

    if diagram:
        caption = ":caption: What this module declares, what those types hold, and what they connect to."
        body = diagram.strip().split("\n")
        if len(body) <= MAXIMUM_INLINE_DIAGRAM:
            lines += ["```{uml}", caption, "", *body, "```", ""]
        else:
            # Long enough that it would bury the prose: it earns a file, and the directive reads it.
            lines += [f"```{{uml}} {document.slug}.puml", caption, "```", ""]
        legend = _diagram_legend(diagram, index)
        if legend:
            lines += legend

    if document.doc.page:
        anchor, title = document.doc.page
        lines += [f"({anchor})=", "", f"{'#' * (level + 1)} {title}", ""]

    namespaces = sorted({entity.namespace for entity in document.entities if entity.namespace})
    if namespaces and not document.is_c_api:
        lines += [f"```{{cpp:namespace}} {namespaces[0]}", "```", ""]

    # A member is documented inside the type that owns it. Declaring it beside the type would put
    # a constructor and its class under the same name at namespace scope, which is a real clash and
    # not just a warning: the reader would see two unrelated entries with one name.
    types = {entity.name for entity in document.entities if entity.kind in {"class", "struct", "enum"}}
    children: dict[str, list[Entity]] = {}
    roots: list[Entity] = []
    for entity in document.entities:
        # A constructor shares its class's name, so identity is the type entity itself — not the
        # name — and only that one is a root.
        is_the_type = entity.kind in {"class", "struct", "enum"} and entity.name == entity.owner
        if entity.owner and entity.owner in types and not is_the_type:
            children.setdefault(entity.owner, []).append(entity)
        else:
            roots.append(entity)

    emitted: set[tuple[str, str, str]] = set()

    def _emit(entity: Entity, depth: int) -> None:
        directive = DIRECTIVES.get((domain, entity.kind))
        if directive is None:
            return
        # Keyed by the declaration, not the name: overloads share a name and are different
        # entities, while a declaration documented twice is the duplicate worth collapsing.
        key = (entity.kind, entity.owner, _signature_for(entity, domain))
        if key in emitted:
            return
        emitted.add(key)

        # Each nesting level needs a longer fence than the one it contains. Nested content is
        # never indented: four spaces inside a fence is an indented code block, which would turn
        # every nested directive into literal text — silently, with no warning from Sphinx.
        fence = "`" * (6 - min(depth, 2))
        indent = ""
        lines.append(f"{indent}{fence}{{{directive}}} {_signature_for(entity, domain)}")
        lines.extend(_render_doc(entity.doc, indent))

        # Unindented, like every other nested directive: an indented fence is a code block.
        enumerator = "c:enumerator" if domain == "c" else "cpp:enumerator"
        for member, description in entity.members:
            lines.append(f"```{{{enumerator}}} {member}")
            if description:
                lines.append(description)
            lines.extend(["```", ""])

        # Rendered as prose rather than a `cpp:member` directive: a field type comes out of a regex
        # and need not be something the C++ domain can parse, and an unparsable signature would fail
        # the whole build under -W.
        documented = [field for field in entity.fields if field[0] == "public" and field[4]]
        if documented:
            lines.extend(["**Fields**", ""])
            for _access, field_type, field_name, _declaration, description in documented:
                lines.append(f"- `{field_name}` (`{field_type}`) — {description}")
            lines.append("")

        # Only a type owns members. A constructor shares its class's name, so looking children up
        # by name alone would nest the whole class inside its own constructor.
        if entity.kind in {"class", "struct", "enum"}:
            for child in children.get(entity.name, []):
                _emit(child, depth + 1)

        lines.extend([f"{indent}{fence}", ""])

    for entity in roots:
        _emit(entity, 0)

    lines.append("")
    return lines


PUML_KINDS = {"class": "class", "struct": "class", "enum": "enum"}

# A member list that runs past this is noise on a diagram; the reference below it has them all.
MAXIMUM_DRAWN_MEMBERS = 8

# Beyond this the edge is labelled with a count rather than a list of member names.
MAXIMUM_EDGE_LABELS = 2

VISIBILITY = {"public": "+", "protected": "#", "private": "-"}


@dataclass
class TypeIndex:
    """Every documented type in the project, so a diagram can reach past its own module."""

    entities: dict[str, Entity]
    owners: dict[str, FileDoc]

    @classmethod
    def Build(cls, documents: list[FileDoc]) -> TypeIndex:
        entities: dict[str, Entity] = {}
        owners: dict[str, FileDoc] = {}
        for document in documents:
            for entity in document.entities:
                if entity.kind in PUML_KINDS and entity.name not in entities:
                    entities[entity.name] = entity
                    owners[entity.name] = document
        return cls(entities=entities, owners=owners)


def _mentions(text: str, names: set[str]) -> set[str]:
    """Which known type names a piece of declaration text refers to."""
    return {name for name in names if re.search(rf"\b{re.escape(name)}\b", text)}


def _relations(drawn: list[Entity], index: TypeIndex) -> tuple[list[tuple[str, str, str, str]], set[str]]:
    """The edges of the diagram, and the outside types they reach.

    Three kinds, because they mean different things: a base class is inheritance, a field is
    composition — the type holds one — and a parameter or return type is a dependency.
    """
    known = set(index.entities)
    local = {entity.name for entity in drawn}
    # Grouped by the pair they connect: five members of one table pointing at the same handle is
    # one relationship, not five edges stacked on top of each other.
    grouped: dict[tuple[str, str, str], list[str]] = {}
    outside: set[str] = set()

    def _record(source: str, target: str, arrow: str, label: str) -> None:
        if target == source or target not in known:
            return
        labels = grouped.setdefault((source, target, arrow), [])
        if label and label not in labels:
            labels.append(label)
        if target not in local:
            outside.add(target)

    for entity in drawn:
        for base in entity.bases:
            _record(entity.name, base, "--|>", "")
        for access, _, field_name, declaration, _description in entity.fields:
            if access != "public":
                continue
            for target in _mentions(declaration, known):
                _record(entity.name, target, "-->", field_name)

    edges: list[tuple[str, str, str, str]] = []
    for (source, target, arrow), labels in sorted(grouped.items()):
        if not labels:
            summary = ""
        elif len(labels) <= MAXIMUM_EDGE_LABELS:
            summary = ", ".join(sorted(labels))
        else:
            summary = f"{len(labels)} members"
        edges.append((source, target, arrow, summary))
    return edges, outside


def _members(entity: Entity) -> list[str]:
    """The lines drawn inside a class box: its fields, then the operations it publishes."""
    public_fields = [field for field in entity.fields if field[0] == "public"]
    lines = [
        f"  {VISIBILITY.get(access, '+')}{name} : {field_type}"
        for access, field_type, name, _declaration, _description in public_fields[:MAXIMUM_DRAWN_MEMBERS]
    ]
    if len(entity.fields) > MAXIMUM_DRAWN_MEMBERS:
        lines.append(f"  .. {len(entity.fields) - MAXIMUM_DRAWN_MEMBERS} more ..")
    return lines


def collect_operations(documents: list[FileDoc]) -> dict[str, list[str]]:
    """The operations each type publishes, keyed by type name."""
    operations: dict[str, list[str]] = {}
    for document in documents:
        for entity in document.entities:
            if entity.kind == "function" and entity.owner and entity.name != entity.owner:
                operations.setdefault(entity.owner, []).append(f"+{entity.name}()")
    return operations


def render_diagram(document: FileDoc, index: TypeIndex, operations: dict[str, list[str]]) -> str | None:
    """A class diagram of one module: what it declares, what those hold, and what they connect to."""
    drawn = [entity for entity in document.entities if entity.kind in PUML_KINDS]
    if not drawn:
        return None

    edges, outside = _relations(drawn, index)
    if not edges and len(drawn) < 2:
        return None

    lines = [
        "@startuml",
        f"title {document.title}",
        "skinparam classAttributeIconSize 0",
        "skinparam shadowing false",
        "hide empty members",
        "",
    ]

    for entity in drawn:
        keyword = PUML_KINDS[entity.kind]
        if entity.kind == "enum":
            body = [f"  {name}" for name, _ in entity.members[:MAXIMUM_DRAWN_MEMBERS]]
        else:
            body = _members(entity)
        body += [f"  {signature}" for signature in operations.get(entity.name, [])[:MAXIMUM_DRAWN_MEMBERS]]
        lines.append(f"{keyword} {entity.name} {{")
        lines += body
        lines.append("}")

    # A type from another module is drawn faintly: it explains this one without competing with it.
    for name in sorted(outside):
        neighbour = index.entities[name]
        lines.append(f"{PUML_KINDS[neighbour.kind]} {name} <<{index.owners[name].title}>> #F5F5F5 {{")
        lines.append("}")

    if outside:
        lines.append("")

    for source, target, arrow, label in edges:
        suffix = f" : {label}" if label else ""
        lines.append(f"{source} {arrow} {target}{suffix}")

    lines += ["", "@enduml", ""]
    return "\n".join(lines)


@dataclass
class Package:
    """One documented package: its prose, its modules, and the types a reader navigates by."""

    key: str
    title: str
    summary: list[str]
    documents: list[FileDoc]


# Where a module belongs. The second component of a module name is the package it lives in, which is
# also the directory that carries the package README.
PACKAGE_TITLES: dict = CONFIG["package_titles"]


def _package_key(document: FileDoc) -> str:
    """The package a translation unit belongs to.

    A partition is grouped with the module it belongs to, so `lightphi.input:samples` lands in the
    same package as `lightphi.input`. A module with a single component is its own package, which is
    what a library named after its one module needs.
    """
    if document.module:
        primary = document.module.split(":", 1)[0]
        parts = primary.split(".")
        return parts[1] if len(parts) > 1 else parts[0]
    return "cxx" if document.path.suffix == ".hpp" else "abi"


def _demote_headings(lines: list[str]) -> list[str]:
    """Move Markdown headings down one level without changing fenced code."""
    result: list[str] = []
    fence: str | None = None
    for line in lines:
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            fence = None if fence == marker else marker
        result.append(f"#{line}" if fence is None and line.startswith("#") else line)
    return result


def _package_summary(root: Path, documents: list[FileDoc]) -> list[str]:
    """The package README's prose, minus its title, so a package page opens with its own words.

    The README is where the reason for a package is already written. Restating it in generated prose
    would be a second copy to keep true, and a worse one.
    """
    directories = [(root / document.path).parent for document in documents]
    candidate = min(directories, key=lambda path: len(path.parts))
    while candidate != root:
        readme = candidate / "README.md"
        if readme.is_file():
            lines = readme.read_text(encoding="utf-8").split("\n")
            body = lines[1:] if lines and lines[0].startswith("# ") else lines
            # Headings drop a level: the package page owns the top one.
            return _demote_headings(body)
        candidate = candidate.parent
    return []


GLANCE_KINDS = frozenset({"class", "struct", "enum"})

# The C domain has no `any` role; a reference names the kind it points at.
C_ROLES = {"class": "c:struct", "struct": "c:struct", "enum": "c:enum"}


def _declared_types(document: FileDoc) -> list[Entity]:
    """The types a module declares, once each, in declaration order."""
    seen: set[str] = set()
    declared: list[Entity] = []
    for entity in document.entities:
        if entity.kind not in GLANCE_KINDS or entity.name in seen:
            continue
        seen.add(entity.name)
        declared.append(entity)
    return declared


def _glance_table(package: Package) -> list[str]:
    """A table of every type the package declares, so a reader can find one without reading the code.

    This is the part a header cannot give you: one line per type, in one place, linked to its entry.
    """
    rows: list[str] = []
    for document in package.documents:
        for entity in _declared_types(document):
            # The C domain has no `any` role, so each kind names its own.
            role = C_ROLES[entity.kind] if document.is_c_api else "cpp:any"
            # A nested type is qualified by the type that owns it as well as by its namespace.
            scope = "::".join(part for part in (entity.namespace, entity.owner) if part)
            qualified = f"{scope}::{entity.name}" if scope else entity.name
            reference = f"{{{role}}}`{entity.name} <{qualified}>`"
            brief = entity.doc.brief or ""
            rows.append(f"| {reference} | `{document.title}` | {brief} |")

    if not rows:
        return []
    return [
        "## Types at a glance",
        "",
        "| Type | Module | What it is |",
        "| --- | --- | --- |",
        *rows,
        "",
    ]


def render_package_page(
    package: Package, repository_url: str, no_diagrams: bool, index: TypeIndex, operations: dict[str, list[str]]
) -> str:
    """One page for one package: why it exists, what it declares, and then the detail."""
    lines = [
        GENERATED_NOTICE,
        "",
        f"# {package.title}",
        "",
        *([*CONFIG["package_note"], ""] if CONFIG.get("package_note") else []),
    ]
    if package.summary:
        lines += [*package.summary, ""]
    lines += _glance_table(package)

    for document in package.documents:
        diagram = None if no_diagrams else render_diagram(document, index, operations)
        lines += render_module_section(document, repository_url, diagram, index)

    return "\n".join(lines).rstrip() + "\n"


def _api_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, pattern in API_SOURCES:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob(pattern)):
            if "build" in path.parts or path in files:
                continue
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repository-url", default=CONFIG["repository_url"])
    parser.add_argument("--output", type=Path, required=True, help="directory to write the API pages into")
    parser.add_argument("--no-diagrams", action="store_true", help="leave the class diagrams out")
    arguments = parser.parse_args()

    root = arguments.root
    api_directory = arguments.output
    if api_directory.exists():
        shutil.rmtree(api_directory)
    api_directory.mkdir(parents=True)

    documents: list[FileDoc] = []
    for path in _api_files(root):
        document = parse_file(path, root)
        if document is None or (not document.entities and not document.doc.brief):
            continue
        documents.append(document)

    grouped: dict[str, list[FileDoc]] = {}
    for document in documents:
        grouped.setdefault(_package_key(document), []).append(document)

    packages = [
        Package(
            key=key,
            title=PACKAGE_TITLES.get(key, key.replace("_", " ").capitalize()),
            summary=_package_summary(root, members),
            documents=members,
        )
        for key, members in sorted(grouped.items(), key=lambda item: item[0])
    ]

    index = TypeIndex.Build(documents)
    operations = collect_operations(documents)

    for package in packages:
        for document in package.documents:
            diagram = None if arguments.no_diagrams else render_diagram(document, index, operations)
            if diagram and len(diagram.strip().split("\n")) > MAXIMUM_INLINE_DIAGRAM:
                (api_directory / f"{document.slug}.puml").write_text(diagram, encoding="utf-8")
        (api_directory / f"{package.key}.md").write_text(
            render_package_page(package, arguments.repository_url, arguments.no_diagrams, index, operations),
            encoding="utf-8",
        )

    entries = "\n".join(package.key for package in packages)
    intro = "\n".join(CONFIG["intro"])
    (api_directory / "index.md").write_text(
        f"""{GENERATED_NOTICE}

# API reference

{intro}

```{{toctree}}
:maxdepth: 1

{entries}
```
""",
        encoding="utf-8",
    )

    entity_count = sum(len(document.entities) for document in documents)
    print(f"[docs] {len(packages)} package pages, {len(documents)} modules, {entity_count} documented entities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
