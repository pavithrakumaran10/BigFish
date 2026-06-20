# Business Rules

Server-side scripts that execute automatically when records are **inserted, updated, deleted, or queried**. Registered under **System Definition > Business Rules**.

## When to Use Each Type

| Type | Runs | Typical Use Case |
|---|---|---|
| `before` | Before DB write | Validation, field population, data transformation |
| `after` | After DB write | Trigger events, create child records, update related records |
| `async` | Background, after commit | Send emails, call REST APIs, heavy processing |
| `display` | Before form loads (read) | Populate display-only scratch pad fields |

## Sub-categories

| Folder | Description |
|---|---|
| `before/` | Before-insert and before-update rules |
| `after/` | After-insert and after-update rules |
| `async/` | Asynchronous (background) processing |

## Best Practices

- Set **Conditions** on the Business Rule record to minimize execution scope.
- Use `current` for the record being modified; `previous` for values before update.
- Never use `GlideRecord.query()` inside a synchronous Business Rule on a large table — move to `async`.
- Use `gs.addInfoMessage()` / `gs.addErrorMessage()` to show messages to the user (before rules only).
- Set `current.setAbortAction(true)` to cancel a save from a before rule.
- Avoid infinite loops: check `current.operation()` and field changes before updating.
