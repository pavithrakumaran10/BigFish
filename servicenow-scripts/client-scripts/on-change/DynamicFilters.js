/**
 * Client Script: DynamicFilters
 * Type:          onChange
 * Table:         /* CONFIGURE: e.g. incident */
 * Field Name:    /* CONFIGURE: e.g. 'assignment_group' */
 * Description:   Dynamically filter reference fields based on other field values.
 *                e.g. filter 'assigned_to' to only show members of the selected group.
 */

function onChange(control, oldValue, newValue, isLoading) {
    if (isLoading) return;

    /* ---- Filter 'assigned_to' to members of the chosen assignment group ---- */
    filterAssignedToByGroup(newValue);

    /* ---- Filter 'cmdb_ci' to CIs matching the chosen category ---- */
    // filterCIByCategory(g_form.getValue('category'));
}

/**
 * Restrict the 'assigned_to' reference field to users in the given group.
 * @param {string} groupSysId
 */
function filterAssignedToByGroup(groupSysId) {
    if (!groupSysId) {
        /* Remove filter: show all users */
        g_form.clearValue('assigned_to');
        g_form.setLookupFilter('assigned_to', '');
        return;
    }

    /* Build encoded query: user must be an active member of the group */
    var filter = 'sys_user_grmember.group=' + groupSysId + '^active=true';
    g_form.clearValue('assigned_to');
    g_form.setLookupFilter('assigned_to', filter);
}

/**
 * Filter 'cmdb_ci' to CIs that match the incident category.
 * @param {string} category
 */
function filterCIByCategory(category) {
    /* CONFIGURE: map category to CI class */
    var categoryClassMap = {
        'hardware': 'cmdb_ci_computer',
        'network':  'cmdb_ci_netgear',
        'software': 'cmdb_ci_appl'
    };

    var ciClass = categoryClassMap[category];
    if (ciClass) {
        g_form.setLookupFilter('cmdb_ci', 'sys_class_name=' + ciClass + '^operational_status=1');
    } else {
        g_form.setLookupFilter('cmdb_ci', 'operational_status=1'); /* default: show operational CIs */
    }
}

/**
 * Apply a dynamic reference qualifier to any reference field.
 * @param {string} fieldName  - the reference field on the form
 * @param {string} encodedQuery - ServiceNow encoded query string
 */
function applyReferenceFilter(fieldName, encodedQuery) {
    g_form.clearValue(fieldName);
    g_form.setLookupFilter(fieldName, encodedQuery);
}
