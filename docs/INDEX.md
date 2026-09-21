# LightRHI documentation

LightRHI is a rendering hardware interface: one portable API over the platform's
native graphics backend. Start with the [README](../README.md) for what it does and
how to build it, then read the document that matches your change.

- [Architecture](ARCHITECTURE.md) — the module layout, CMake targets, backend
  selection, and the shader pipeline. Read this before moving a responsibility or
  adding a dependency.
- [API guidelines](API_GUIDELINES.md) — the rules the public API follows. Read this
  before changing an exported contract.
- [Package map](PACKAGE_MAP.md) — the public module, platform backends, tooling,
  examples, and test ownership.
- [Developer commands](DEVELOPER_COMMANDS.md) — standalone configure, build,
  test, documentation, and sanitizer commands.
- [Test strategy](TEST_STRATEGY.md) — smoke and GPU-dependent coverage, labels,
  and skip expectations.
- [Sanitizer support](SANITIZER_SUPPORT.md) — supported standalone sanitizer
  configurations and their limits.

```{toctree}
:hidden:

ARCHITECTURE
API_GUIDELINES
PACKAGE_MAP
DEVELOPER_COMMANDS
TEST_STRATEGY
SANITIZER_SUPPORT
```

```{toctree}
:hidden:
:caption: API reference

api/index
```
