/**
 * Business Rule: BeforeInsertValidation
 * Table:         /* CONFIGURE: e.g. incident */
 * When:          before
 * Insert:        true  |  Update: false  |  Delete: false  |  Query: false
 * Order:         100
 * Description:   Validates data integrity before a new record is inserted.
 *                Aborts the insert and shows an error message if validation fails.
 *
 * Condition (on the BR record): /* CONFIGURE e.g. current.category == 'hardware' */
 */

(function executeRule(current, previous) {

    /* ---- 1. Mandatory field check ---- */
    var requiredFields = ['short_description', 'category', 'caller_id'];
    for (var i = 0; i < requiredFields.length; i++) {
        if (!current[requiredFields[i]].nil()) continue;
        current.setAbortAction(true);
        gs.addErrorMessage('Field is required: ' + requiredFields[i]);
        return;
    }

    /* ---- 2. Due date must be in the future ---- */
    if (!current.due_date.nil()) {
        var due  = new GlideDateTime(current.due_date.toString());
        var now  = new GlideDateTime();
        if (due.before(now)) {
            current.setAbortAction(true);
            gs.addErrorMessage('Due date must be set to a future date and time.');
            return;
        }
    }

    /* ---- 3. Prevent duplicate: same caller + same short_description open in last 24h ---- */
    /* CONFIGURE: adjust table and field names as needed */
    var gr = new GlideRecord('incident');
    gr.addQuery('caller_id', current.caller_id.toString());
    gr.addQuery('short_description', current.short_description.toString());
    gr.addQuery('state', 'NOT IN', '6,7');  // not Resolved or Closed
    var cutoff = new GlideDateTime();
    cutoff.addDaysUTC(-1);
    gr.addQuery('sys_created_on', '>=', cutoff.getValue());
    gr.setLimit(1);
    gr.query();
    if (gr.next()) {
        current.setAbortAction(true);
        gs.addErrorMessage(
            'A similar open incident already exists: <a href="incident.do?sys_id=' +
            gr.getUniqueValue() + '">' + gr.getValue('number') + '</a>'
        );
        return;
    }

    /* ---- 4. Auto-set priority from impact + urgency ---- */
    if (!current.impact.nil() && !current.urgency.nil()) {
        current.priority = calculatePriority(
            parseInt(current.impact.toString()),
            parseInt(current.urgency.toString())
        );
    }

})(current, previous);

/**
 * Calculate priority from impact (1-3) and urgency (1-3).
 * Matrix: 1+1=1 (Critical), 1+2 or 2+1=2 (High), etc.
 */
function calculatePriority(impact, urgency) {
    var matrix = {
        '1_1': 1, '1_2': 2, '1_3': 3,
        '2_1': 2, '2_2': 3, '2_3': 4,
        '3_1': 3, '3_2': 4, '3_3': 5
    };
    return matrix[impact + '_' + urgency] || 4;
}
