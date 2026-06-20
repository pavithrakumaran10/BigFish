/**
 * Client Script: ReferenceFieldLookup
 * Type:          onChange
 * Table:         /* CONFIGURE: your target table */
 * Field Name:    /* CONFIGURE: the reference field being watched, e.g. 'cmdb_ci' */
 * Description:   Auto-populate fields based on a reference field selection.
 *                Uses GlideAjax to fetch related data server-side.
 *
 * Requires: A Client Callable Script Include with the lookup method.
 */

function onChange(control, oldValue, newValue, isLoading) {
    if (isLoading) return;

    if (!newValue) {
        clearRelatedFields();
        return;
    }

    fetchCIDetails(newValue);
}

/**
 * Call server-side Script Include to get Configuration Item details.
 * @param {string} ciSysId
 */
function fetchCIDetails(ciSysId) {
    var ga = new GlideAjax('CommonUtils'); /* CONFIGURE: your Script Include name */
    ga.addParam('sysparm_name',   'ajaxGetCIInfo'); /* CONFIGURE: method name */
    ga.addParam('sysparm_ci_id',  ciSysId);
    ga.getXML(onCILoaded);
}

/**
 * Callback: populate form fields with CI data returned from the server.
 * @param {XMLHttpRequest} response
 */
function onCILoaded(response) {
    var answer = response.responseXML.documentElement.getAttribute('answer');
    if (!answer) { clearRelatedFields(); return; }

    try {
        var ci = JSON.parse(answer);
        /* CONFIGURE: map CI properties to form fields */
        if (ci.support_group)    g_form.setValue('assignment_group', ci.support_group,    ci.support_group_display);
        if (ci.location)         g_form.setValue('location',         ci.location,          ci.location_display);
        if (ci.owned_by)         g_form.setValue('u_ci_owner',       ci.owned_by,          ci.owned_by_display);
    } catch (e) {
        if (window.console) console.error('ReferenceFieldLookup: parse error', e);
    }
}

/** Clear fields that were auto-populated by a previous selection. */
function clearRelatedFields() {
    /* CONFIGURE: match the fields you populate in onCILoaded */
    g_form.clearValue('assignment_group');
    g_form.clearValue('location');
    g_form.clearValue('u_ci_owner');
}
