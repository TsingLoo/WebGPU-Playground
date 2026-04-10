# Project Rules & Documentation Sync

1. **Architecture Sync**: Any modifications to the rendering pipelines, shaders, RenderGraph, or architecture MUST be reflected directly in the `doc/tech_report.md` and the `doc/structure_and_flowchart.md` files.
2. **Mermaid Flowchart Syntax Requirement**: To guarantee 100% compatibility with the built-in VS Code markdown previewer, all Mermaid charts must strictly adhere to the following baseline syntax:
   - **No Subgraph Titles**: Use the format `subgraph PassExecution` (alphanumeric only). Heavily avoid the `subgraph ID [Title]` formatting.
   - **No Edge Labels**: Avoid `-->|Text|` labels as they crash older parsers when spaces or special characters are present. Use markdown text outside the chart to explain conditional routing.
   - **Standard Nodes Only**: Enforce the use of standard square brackets `[Text Node]` for all nodes. Do not use quotes `["Text"]`, diamond selectors `{Text}`, or database cylinders `[(Text)]`.
   - **No Special Characters**: Exclude characters such as `<`, `>`, `&`, `/`, `(`, `)`, and `"` inside node definitions.
   - **No Indentation**: Do not indent the node entries inside subgraph blocks.
