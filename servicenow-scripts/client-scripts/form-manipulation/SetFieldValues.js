/**
 * Client Script: SetFieldValues
 * Type:          onChange
 * Table:         /* CONFIGURE: e.g. incident */
 * Field Name:    /* CONFIGURE: e.g. category */
 * Description:   Auto-populate related fields when a reference or choice field changes.
 *
 * Common patterns:
 *   - Set subcategory choices when category changes.
 *   - Populate contact info when caller is selected.
 *   - Set assignment group from configuration item.
 */

function onChange(control, oldValue, newValue, isLoading) {
    if (isLoading) return;
    if (!newValue) return;

    /* ---- Pattern 1: populate fields from a reference field lookup via GlideAjax ---- */
    populateCallerInfo(newValue);

    /* ---- Pattern 2: set a value directly on another field ---- */
    // g_form.setValue('assignment_group', 'group_sys_id_here');
}

/**
 * Fetch user details via GlideAjax and populate form fields.
 * Requires a Client Callable Script Include 'UserGroupUtils' with ajaxGetUserInfo method.
 */
function populateCallerInfo(callerSysId) {
    var ga = new GlideAjax('UserGroupUtils');
    ga.addParam('sysparm_name',    'ajaxGetUserInfo');  /* CONFIGURE: SI method name */
    ga.addParam('sysparm_user_id', callerSysId);
    ga.getXML(function(response) {
        var answer = response.responseXML.documentElement.getAttribute('answer');
        if (!answer) return;
        try {
            var user = JSON.parse(answer);
            g_form.setValue('location',  user.location  || '');
            g_form.setValue('department', user.department || '');
            /* CONFIGURE: add more field mappings as needed */
        } catch (e) {
            if (window.console) console.error('populateCallerInfo parse error', e);
        }
    });
}

/* ---- Pattern 3: cascading choice list (category → subcategory) ---- */
function onCategoryChange(control, oldValue, newValue, isLoading) {
    if (isLoading) return;
    /* Clear subcategory when category changes */
    g_form.clearValue('subcategory');

    /* CONFIGURE: map category values to allowed subcategory values */
    var subMap = {
        'hardware':  ['Laptop', 'Desktop', 'Printer'],
        'software':  ['OS', 'Application', 'Security'],
        'network':   ['VPN', 'Wi-Fi', 'LAN']
    };

    var subs = subMap[newValue] || [];
    var field = g_form.getField('subcategory');
    if (!field) return;

    /* Rebuild the choice list */
    var select = field.getSelectElement();
    while (select.options.length > 1) select.remove(1); // keep blank first option
    for (var i = 0; i < subs.length; i++) {
        var opt = document.createElement('option');
        opt.value = subs[i].toLowerCase().replace(/ /g, '_');
        opt.text  = subs[i];
        select.add(opt);
    }
}
