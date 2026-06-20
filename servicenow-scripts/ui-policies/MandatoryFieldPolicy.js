/**
 * UI Policy Equivalent - Client Script: MandatoryFieldPolicy
 * Type:    onLoad + onChange
 * Table:   /* CONFIGURE: e.g. incident */
 * Description:
 *   Enforce context-sensitive mandatory fields that go beyond static UI Policy.
 *
 *   Rules:
 *   - 'work_notes' is mandatory when state changes to 'In Progress'.
 *   - 'close_code' and 'close_notes' are mandatory when state = 'Resolved'.
 *   - 'u_business_justification' is mandatory when impact = 1 (High Business Impact).
 */

function onLoad() {
    enforceMandatoryFields();
}

function onChange(control, oldValue, newValue, isLoading) {
    if (isLoading) return;
    enforceMandatoryFields();
}

function enforceMandatoryFields() {
    var state  = g_form.getValue('state');
    var impact = g_form.getValue('impact');

    /* Work notes mandatory when moving to In Progress */
    g_form.setMandatory('work_notes', state === '2'); /* CONFIGURE: '2' = In Progress */

    /* Resolution fields mandatory when Resolved */
    var isResolved = (state === '6');
    g_form.setMandatory('close_code',  isResolved);
    g_form.setMandatory('close_notes', isResolved);
    g_form.setMandatory('resolved_by', isResolved);

    /* Business justification mandatory for high-impact records */
    g_form.setMandatory('u_business_justification', impact === '1'); /* CONFIGURE */

    /* Additional read-only rules */
    g_form.setReadOnly('resolved_at',  isResolved); /* prevent manual override */
    g_form.setReadOnly('resolved_by',  isResolved);
}
