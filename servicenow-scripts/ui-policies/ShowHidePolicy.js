/**
 * UI Policy Equivalent - Client Script: ShowHidePolicy
 * Type:    onLoad + onChange
 * Table:   /* CONFIGURE: e.g. sc_request_item */
 * Description:
 *   Implements show/hide logic equivalent to a UI Policy, but with additional
 *   flexibility for complex multi-condition scenarios.
 *
 * Simpler cases: use System Definition > UI Policies directly (no code needed).
 * Use this script when the UI Policy conditions cannot express the logic.
 *
 * Rule applied:
 *   IF category = 'hardware' AND urgency <= 2
 *   THEN show: model, quantity, delivery_address
 *        hide: software_version, license_key
 *
 *   ELSE
 *        hide: model, quantity, delivery_address
 *        show: software_version, license_key
 */

function onLoad() {
    applyFieldVisibility();
}

function onChange(control, oldValue, newValue, isLoading) {
    if (isLoading) return;
    /* Only re-evaluate when the relevant fields change */
    var watchedFields = ['category', 'urgency'];
    if (watchedFields.indexOf(control) === -1) return;
    applyFieldVisibility();
}

function applyFieldVisibility() {
    var category = g_form.getValue('category');
    var urgency  = parseInt(g_form.getValue('urgency') || '3', 10);
    var isHardware = (category === 'hardware') && (urgency <= 2);

    /* Hardware fields */
    var hardwareFields = ['u_model', 'u_quantity', 'u_delivery_address']; /* CONFIGURE */
    /* Software fields */
    var softwareFields = ['u_software_version', 'u_license_key'];          /* CONFIGURE */

    setFieldsVisible(hardwareFields, isHardware);
    setFieldsVisible(softwareFields, !isHardware);
    setFieldsMandatory(hardwareFields, isHardware);
    setFieldsMandatory(softwareFields, !isHardware);
}

function setFieldsVisible(fieldNames, visible) {
    for (var i = 0; i < fieldNames.length; i++) {
        g_form.setVisible(fieldNames[i], visible);
    }
}

function setFieldsMandatory(fieldNames, mandatory) {
    for (var i = 0; i < fieldNames.length; i++) {
        g_form.setMandatory(fieldNames[i], mandatory);
    }
}
