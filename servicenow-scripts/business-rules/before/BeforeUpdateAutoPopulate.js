/**
 * Business Rule: BeforeUpdateAutoPopulate
 * Table:         /* CONFIGURE: e.g. incident */
 * When:          before
 * Insert:        false  |  Update: true  |  Delete: false  |  Query: false
 * Order:         100
 * Condition:     current.state.changesTo(6)   /* Triggers when state changes to Resolved */
 * Description:   Auto-populate fields when specific conditions are met during update.
 */

(function executeRule(current, previous) {

    /* ---- Set resolved_at and resolved_by when state moves to Resolved ---- */
    if (current.state.changesTo('6')) { /* CONFIGURE: '6' = Resolved */
        if (current.resolved_at.nil()) {
            current.resolved_at = new GlideDateTime();
        }
        if (current.resolved_by.nil()) {
            current.resolved_by = gs.getUserID();
        }
        /* Validate that close_code is provided */
        if (current.close_code.nil()) {
            current.setAbortAction(true);
            gs.addErrorMessage('Close Code is required to resolve an incident.');
            return;
        }
    }

    /* ---- Clear resolved fields if state is moved back to active ---- */
    if (previous.state.toString() === '6' && current.state.toString() !== '6' && current.state.toString() !== '7') {
        current.resolved_at = null;
        current.resolved_by = null;
        gs.addInfoMessage('Incident re-opened: resolved fields have been cleared.');
    }

    /* ---- Auto-assign to self if 'Assigned' state is set without an assignee ---- */
    if (current.state.changesTo('2') && current.assigned_to.nil()) { /* CONFIGURE: '2' = Assigned */
        current.assigned_to = gs.getUserID();
        gs.addInfoMessage('This incident has been auto-assigned to you.');
    }

    /* ---- Stamp last_updated_by (custom field) ---- */
    /* CONFIGURE: ensure u_last_updated_by field exists on the table */
    // current.u_last_updated_by = gs.getUserID();

})(current, previous);
