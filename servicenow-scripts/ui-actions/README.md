# UI Actions

UI Actions are buttons, links, and menu items on forms and lists. Registered under **System Definition > UI Actions**.

## Types

| Type | Location | Notes |
|---|---|---|
| Form button | Top of form | Most common; set **Form button = true** |
| Form link | Bottom of related lists | Set **Form link = true** |
| Form context menu | Right-click on form | Set **Form context menu = true** |
| List button | Top of list | Set **List button = true** |
| List link | Below list | Set **List link = true** |
| List context menu | Right-click on list row | Set **List context menu = true** |

## Anatomy of a UI Action Script

```javascript
// Client-side check (runs in browser before calling server)
function onClick() {
    if (!confirm('Are you sure?')) return false; // cancel
    gsftSubmit(null, g_form.getFormElement(), 'action_name_here');
}

// Server-side processing (Action name must match gsftSubmit call)
if (typeof window === 'undefined') {
    // Server-side code here
    current.setValue('state', '6');
    current.update();
    action.setRedirectURL(current);
}
```

## Best Practices

- Always confirm destructive actions with a client-side `confirm()` dialog.
- Use `action.setRedirectURL(current)` to return to the same record after processing.
- Use **Condition** field on the record to show/hide based on field values or roles.
- For complex server logic, delegate to a Script Include method.
