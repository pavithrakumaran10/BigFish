# Client Scripts

Client-side JavaScript that executes in the user's browser. Registered under **System Definition > Client Scripts**.

## Script Types

| Type | Trigger | Use Case |
|---|---|---|
| `onLoad` | When a form is loaded | Set defaults, show/hide fields on open |
| `onChange` | When a specific field value changes | Conditional logic, dynamic lookups |
| `onSubmit` | When the form is submitted | Final validation before save |
| `onCellEdit` | When a list cell is edited | Validate inline edits |

## Sub-categories

| Folder | Description |
|---|---|
| `form-manipulation/` | Show/hide fields, set field values, make fields read-only |
| `validation/` | Field-level and form-level validation |
| `reference-fields/` | Auto-populate from reference lookups |
| `on-change/` | Dynamic choice lists, field filters, cascading behavior |

## Important Notes

- Client scripts run in the browser — they cannot use `GlideRecord` directly for heavy queries.
- Use **GlideAjax** to call a `Client Callable` Script Include for server-side data.
- Wrap all DOM access in `typeof g_form !== 'undefined'` guards when testing outside ServiceNow.
- For Service Portal, use AngularJS controllers or Widget Client Scripts instead.
- Set the **Table** field on the Client Script record to scope it to the correct form.
