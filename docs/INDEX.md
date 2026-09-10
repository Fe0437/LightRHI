# LightRHI documentation

LightRHI is a rendering hardware interface: one portable API over the platform's
native graphics backend. Start with the [README](../README.md) for what it does and
how to build it, then read the document that matches your change.

- [Architecture](ARCHITECTURE.md) — the module layout, CMake targets, backend
  selection, and the shader pipeline. Read this before moving a responsibility or
  adding a dependency.
- [API guidelines](API_GUIDELINES.md) — the rules the public API follows. Read this
  before changing an exported contract.

```{toctree}
:hidden:

ARCHITECTURE
API_GUIDELINES
```

```{toctree}
:hidden:
:caption: API reference

api/index
```
