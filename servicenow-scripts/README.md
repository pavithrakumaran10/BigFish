# ServiceNow Developer Scripts Repository

A comprehensive collection of reusable scripts for ServiceNow developers, organized by category. All scripts are production-ready, well-commented, and designed to be easily adapted to your instance.

## Repository Structure

```
servicenow-scripts/
├── script-includes/          # Server-side reusable JS libraries
│   ├── utils/                # String, Date, Array helpers
│   ├── glide-record/         # GlideRecord CRUD helpers
│   ├── rest-api/             # Outbound REST utilities
│   ├── user-group/           # User & Group utilities
│   ├── notification/         # Email & Notification utilities
│   └── common/               # General-purpose utilities
├── client-scripts/           # Client-side form scripts
│   ├── form-manipulation/    # Show/hide fields, set values
│   ├── validation/           # Field & form validation
│   ├── reference-fields/     # Reference field helpers
│   └── on-change/            # Dynamic filters & behaviors
├── business-rules/           # Server-side business logic
│   ├── before/               # Before insert/update rules
│   ├── after/                # After insert/update rules
│   └── async/                # Asynchronous background rules
├── ui-actions/               # Buttons and list actions
│   ├── form/                 # Form-level UI actions
│   └── list/                 # List-level bulk actions
├── ui-policies/              # Field visibility & mandatory rules
├── code-snippets/            # Quick-reference copy-paste snippets
│   ├── glide-record/         # GlideRecord patterns
│   ├── rest-api/             # REST API examples
│   ├── json-xml/             # JSON/XML handling
│   ├── date-time/            # Date and time operations
│   └── miscellaneous/        # Attachments, ACLs, workflows
└── flow-designer/            # Flow Designer action & subflow scripts
    ├── custom-actions/
    └── subflows/
```

## Quick Start

1. Browse to the category matching your use case.
2. Copy the script content.
3. Create the appropriate record in your ServiceNow instance (Script Include, Client Script, etc.).
4. Replace placeholder values (table names, field names, sys_ids) with your instance-specific values.
5. Test in a sub-production instance before promoting to production.

## Categories at a Glance

| Category | Where to create in ServiceNow | Runs on |
|---|---|---|
| Script Includes | System Definition > Script Includes | Server |
| Client Scripts | System Definition > Client Scripts | Browser |
| Business Rules | System Definition > Business Rules | Server |
| UI Actions | System Definition > UI Actions | Server / Browser |
| UI Policies | System Definition > UI Policies | Browser |
| Flow Designer | Process Automation > Flow Designer | Server |

## Conventions

- Every script has a header block: **Purpose**, **Table**, **Trigger**, **Author**, **Version**.
- Placeholder values are wrapped in `/* CONFIGURE: ... */` comments.
- All server-side scripts use `gs.log` / `gs.error` for logging — swap for `gs.debug` in production.
- Client scripts use `console.log` wrapped in `if (window.console)` guards for IE compatibility.

## Contributing

1. Follow the existing file structure and naming conventions.
2. Add a descriptive header comment block to every new script.
3. Test in a sub-production instance before committing.
4. Open a PR with a description of what the script does and which table/application it targets.

## License

MIT — free to use, modify, and distribute in your ServiceNow projects.
