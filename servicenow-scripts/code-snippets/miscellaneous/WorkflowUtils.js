/**
 * Snippet Collection: Workflow & Flow Utilities
 * Context: Business Rules, Script Includes, Background Scripts
 * Description: Trigger workflows, check states, and interact with Flow Designer.
 */

/* ---- Trigger a legacy Workflow on a record ---- */
function triggerWorkflow(tableName, recordSysId, workflowName) {
    var wf = new Workflow();
    var gr = new GlideRecord(tableName);
    if (!gr.get(recordSysId)) {
        gs.error('WorkflowUtils: record not found - ' + tableName + ' / ' + recordSysId);
        return null;
    }
    return wf.startFlow(workflowName, gr, 'start'); /* CONFIGURE: workflow name */
}

/* ---- Cancel all active workflow contexts for a record ---- */
function cancelWorkflows(tableName, recordSysId) {
    var gr = new GlideRecord('wf_context');
    gr.addQuery('id',         recordSysId);
    gr.addQuery('table',      tableName);
    gr.addQuery('state', 'IN', 'executing,waiting,paused_for_timer,paused_for_approval');
    gr.query();
    var wf = new Workflow();
    while (gr.next()) {
        wf.cancel(gr);
    }
}

/* ---- Trigger a Flow Designer flow ---- */
function triggerFlow(flowName, inputs) {
    /* CONFIGURE: flowName e.g. 'global.my_flow' */
    try {
        sn_fd.FlowAPI.startFlow(
            flowName,
            null,   /* current record (optional) */
            'now',  /* schedule */
            inputs  /* { param_name: value } */
        );
    } catch (ex) {
        gs.error('triggerFlow failed for ' + flowName + ': ' + ex.message);
    }
}

/* ---- Trigger a Flow Designer subflow ---- */
function triggerSubflow(subflowName, inputs) {
    /* CONFIGURE: subflowName e.g. 'global.my_subflow' */
    try {
        sn_fd.FlowAPI.startSubflow(
            subflowName,
            null,
            'now',
            inputs
        );
    } catch (ex) {
        gs.error('triggerSubflow failed for ' + subflowName + ': ' + ex.message);
    }
}

/* ---- Check if a workflow is running for a record ---- */
function isWorkflowRunning(tableName, recordSysId) {
    var gr = new GlideRecord('wf_context');
    gr.addQuery('id',    recordSysId);
    gr.addQuery('table', tableName);
    gr.addQuery('state', 'IN', 'executing,waiting,paused_for_timer,paused_for_approval');
    gr.setLimit(1);
    gr.query();
    return gr.next();
}

/* ---- Log a workflow activity note (in wf_log) ---- */
function logWorkflowNote(contextSysId, note) {
    var log = new GlideRecord('wf_log');
    log.initialize();
    log.setValue('context', contextSysId);
    log.setValue('log',     note);
    log.insert();
}
