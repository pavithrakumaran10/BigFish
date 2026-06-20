/**
 * Business Rule: AsyncProcessing
 * Table:         /* CONFIGURE: e.g. incident */
 * When:          async
 * Insert:        true  |  Update: true  |  Delete: false  |  Query: false
 * Order:         100
 * Condition:     current.state.changes()
 * Description:   Perform background (non-blocking) processing after a record is saved.
 *                Async rules run in a separate thread after the transaction commits.
 *                Safe for REST API calls, heavy GlideRecord queries, and long-running tasks.
 */

(function executeRule(current, previous) {

    /* ---- Call an external REST API to sync data ---- */
    syncToExternalSystem(current);

    /* ---- Create audit/history log entry in a custom table ---- */
    createAuditEntry(current, previous);

    /* ---- Recalculate SLA metrics ---- */
    // recalculateSLABreaches(current.getUniqueValue());

})(current, previous);

/**
 * Push record data to an external system via REST.
 * @param {GlideRecord} record
 */
function syncToExternalSystem(record) {
    try {
        var payload = {
            number:        record.getValue('number'),
            state:         record.getValue('state'),
            description:   record.getValue('short_description'),
            updated_at:    record.getValue('sys_updated_on'),
            external_ref:  record.getValue('u_external_ref') /* CONFIGURE: your field */
        };

        var rm = new sn_ws.RESTMessageV2(); /* CONFIGURE: use named REST Message if available */
        rm.setEndpoint(gs.getProperty('u_external_api_url')); /* CONFIGURE: system property for URL */
        rm.setHttpMethod('POST');
        rm.setRequestHeader('Content-Type', 'application/json');
        rm.setRequestHeader('Authorization', 'Bearer ' + gs.getProperty('u_external_api_token'));
        rm.setRequestBody(JSON.stringify(payload));
        rm.setHttpTimeout(15000);

        var response = rm.execute();
        if (response.getStatusCode() !== 200 && response.getStatusCode() !== 201) {
            gs.error('AsyncProcessing.syncToExternalSystem failed: HTTP ' +
                response.getStatusCode() + ' | ' + response.getBody());
        }
    } catch (ex) {
        gs.error('AsyncProcessing.syncToExternalSystem exception: ' + ex.message);
    }
}

/**
 * Write a state-change audit entry to a custom log table.
 * @param {GlideRecord} current
 * @param {GlideRecord} previous
 */
function createAuditEntry(current, previous) {
    /* CONFIGURE: ensure 'u_incident_audit_log' table exists */
    if (!current.state.changes()) return;

    try {
        var log = new GlideRecord('u_incident_audit_log');
        log.initialize();
        log.setValue('u_incident',      current.getUniqueValue());
        log.setValue('u_previous_state', previous.getValue('state'));
        log.setValue('u_new_state',      current.getValue('state'));
        log.setValue('u_changed_by',     gs.getUserID());
        log.setValue('u_changed_at',     new GlideDateTime().getValue());
        log.insert();
    } catch (ex) {
        gs.error('AsyncProcessing.createAuditEntry exception: ' + ex.message);
    }
}
