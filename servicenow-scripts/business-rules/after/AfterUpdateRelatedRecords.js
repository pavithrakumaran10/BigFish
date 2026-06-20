/**
 * Business Rule: AfterUpdateRelatedRecords
 * Table:         /* CONFIGURE: e.g. incident */
 * When:          after
 * Insert:        false  |  Update: true  |  Delete: false  |  Query: false
 * Order:         100
 * Condition:     current.state.changesTo(6)  /* When resolved */
 * Description:   Create or update related records after the main record is saved.
 *                e.g. create a Problem, update a Change, close child tasks.
 */

(function executeRule(current, previous) {

    /* ---- Close all child tasks when parent incident is resolved ---- */
    if (current.state.changesTo('6')) {
        closeChildTasks(current.getUniqueValue());
    }

    /* ---- Update a linked Problem record ---- */
    if (!current.problem_id.nil() && current.state.changesTo('6')) {
        updateLinkedProblem(current.problem_id.toString(), current);
    }

    /* ---- Cascade assignment group change to child tasks ---- */
    if (current.assignment_group.changes()) {
        reassignChildTasks(
            current.getUniqueValue(),
            current.assignment_group.toString()
        );
    }

})(current, previous);

/**
 * Close all active task records whose parent is this incident.
 * @param {string} incidentSysId
 */
function closeChildTasks(incidentSysId) {
    /* CONFIGURE: table and field names for your child task relationship */
    var gr = new GlideRecord('task');
    gr.addQuery('parent', incidentSysId);
    gr.addQuery('state', 'NOT IN', '3,4'); // not Closed or Cancelled
    gr.query();
    while (gr.next()) {
        gr.setValue('state', '3'); /* CONFIGURE: '3' = Closed */
        gr.setValue('close_notes', 'Auto-closed: parent incident resolved.');
        gr.update();
    }
}

/**
 * Update a linked Problem record with resolution info.
 * @param {string} problemSysId
 * @param {GlideRecord} incident
 */
function updateLinkedProblem(problemSysId, incident) {
    var problem = new GlideRecord('problem');
    if (!problem.get(problemSysId)) return;

    /* Only update if the problem isn't already resolved/closed */
    if (problem.getValue('state') === '4' || problem.getValue('state') === '5') return;

    problem.setValue('work_notes',
        'Linked incident ' + incident.getValue('number') + ' was resolved.\n' +
        'Resolution: ' + incident.getValue('close_notes')
    );
    problem.update();
}

/**
 * Push assignment group change to all open child tasks.
 * @param {string} parentSysId
 * @param {string} groupSysId
 */
function reassignChildTasks(parentSysId, groupSysId) {
    var gr = new GlideRecord('task');
    gr.addQuery('parent', parentSysId);
    gr.addQuery('state', 'NOT IN', '3,4');
    gr.query();
    while (gr.next()) {
        gr.setValue('assignment_group', groupSysId);
        gr.update();
    }
}
