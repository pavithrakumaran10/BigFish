# UI Policies

UI Policies control field **visibility**, **mandatory** status, and **read-only** state on forms without writing JavaScript. Registered under **System Definition > UI Policies**.

## When to Use UI Policies vs Client Scripts

| Approach | Best When |
|---|---|
| UI Policy | Simple if/then logic on one or two fields |
| Client Script | Complex conditions, multi-field logic, GlideAjax calls |

## UI Policy Structure

- **Condition** — When this evaluates to true, the policy actions run.
- **Reverse if false** — When condition becomes false, reverse the actions.
- **On load** — Also evaluate the condition when the form first loads.
- **UI Policy Actions** — Field-level rules: mandatory, visible, read-only.

## Script Files in This Folder

The `.js` files here contain the equivalent **Client Script** implementation for cases where UI Policy UI alone is insufficient — for example, when:
- The condition involves server-side data not available on the form.
- You need to set a field *value*, not just its state.
- You need to act on more than ~5 fields efficiently.
