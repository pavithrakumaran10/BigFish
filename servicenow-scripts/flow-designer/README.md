# Flow Designer Scripts

Scripts used within **Flow Designer** flows, subflows, and actions. Registered under **Process Automation > Flow Designer**.

## Types of Scripts in Flow Designer

| Type | Location | Purpose |
|---|---|---|
| **Action Script Step** | Inside a Custom Action | Run server-side JS within a flow step |
| **Flow Script** | Inside a Flow's Script step | Inline script within a flow |
| **Subflow** | Standalone reusable unit | Encapsulate common logic called by multiple flows |
| **Decision Table** | Flow branch logic | Condition-based routing |

## Sub-categories

| Folder | Description |
|---|---|
| `custom-actions/` | Reusable action scripts callable from any flow |
| `subflows/` | Example subflow script configurations |

## Tips

- Use **Action Script Steps** rather than inline Flow Scripts for reusability.
- Define inputs and outputs explicitly — Flow Designer is strongly typed.
- Test actions in isolation using the **Test** button before using in flows.
- For complex logic, delegate to a Script Include and call it from the action script.
- Use `fd_data` (Flow Data) to access flow-level variables inside a script step.

## Script Step Template

```javascript
(function execute(inputs, outputs) {
    // inputs.your_input_name  <- defined in the action's Inputs section
    // outputs.your_output_name = value;  <- defined in Outputs section

    var sysId = inputs.record_sys_id;  // CONFIGURE
    var gr    = new GlideRecord('incident');
    if (gr.get(sysId)) {
        outputs.number      = gr.getValue('number');
        outputs.description = gr.getValue('short_description');
        outputs.success     = true;
    } else {
        outputs.success = false;
        outputs.error   = 'Record not found: ' + sysId;
    }
})(inputs, outputs);
```
