# CLAUDE Design Guidelines

This document defines the design file organization and formatting standards for the fintech project.

---

## Design File Organization

### Directory Structure

Design files are stored in `.arch/`. Your goal is to maintain them as code changes. This is one of your priorities, to maintain the state of the codebase as it evolves.

They are organized as follows:

```
.arch/
  src/
    README.md                 # System-level design overview
    ingestion/
      README.md               # Module design: Data Ingestion
    db/
      README.md               # Module design: Database / Storage
    normalize/
      README.md               # Module design: Financial Normalization & Quality
    valuation/
      README.md               # Module design: Valuation / Model Optimization
    strategy/
      README.md               # Module design: Portfolio Construction
    evaluation/
      README.md               # Module design: Backtesting / Evaluation
    api/
      README.md               # Module design: API Layer
    reporting/
      README.md               # Module design: Reporting
    providers/
      README.md               # Module design: External Data Providers
```

### Naming Conventions

- **Main folder**: `src` (the primary codebase folder)
- **Module folders**: Subfolders of the main folder (e.g., `ingestion/`, `valuation/`)
- **Design files**: `README.md` within each `.arch/<path>/` directory

---

## Design File Template

Each module design file should contain the following sections:

```markdown
# <Module Name>

## Responsibilities
- What this module is responsible for (bulleted list)

## Inputs
- Data/configuration this module consumes

## Outputs
- Data/artifacts this module produces

## Dependencies
- External packages and internal modules this depends on

## Folder Structure
- Expected file organization within the module

## Design Decisions
- Key architectural choices and their rationale
- Trade-offs considered
- Items requiring review (marked with ⚠️)

## Constraints
- Hard constraints this module must respect
- What this module must NOT do
```

---

## Design Review Markers

Use the following markers to flag items for review:

- **⚠️ NEEDS REVIEW**: Design decision that requires stakeholder input
- **📌 TODO**: Known gap that needs to be addressed
- **⚡ CONSTRAINT**: Hard constraint that must be enforced
- **💡 ALTERNATIVE**: Alternative approach considered but not chosen

---

## Module Dependency Rules

1. Modules should have clear, unidirectional dependencies
2. Lower-level modules (db, providers) should not depend on higher-level modules (strategy, evaluation)
3. Cross-cutting concerns (logging, config) should be in shared utilities

---

## Source File Reference

This structure is derived from:
- `LLMEM.md` - Original design organization guidelines
- `.arch/src/README.md` - System requirements and module specifications

---

## Python Coding Standards

All Python code in this project **MUST** follow these requirements:

### Type Annotations
- **All functions** must have complete type hints (parameters and return types)
- Use `typing` module for complex types (`List`, `Dict`, `Optional`, `Callable`, etc.)
- Use `pydantic`'s `BaseModel` for data structures (dataclasses are deprecated)

### Linting & Formatting
- **Ruff** for linting (or `flake8` + `isort`)
- **Black** for formatting (line length: 88)
- **mypy** for static type checking (strict mode recommended)

### Best Practices
- Follow PEP 8 naming conventions
- Docstrings for all public functions/classes (Google or NumPy style)
- No `# type: ignore` without explicit justification
- Prefer composition over inheritance
- Use explicit imports (no `from module import *`)
