/**
 * Snippet Collection: Scheduled Job Scripts
 * Context: System Definition > Scheduled Jobs (sys_trigger)
 * Description: Example scripts for common recurring maintenance and reporting tasks.
 *
 * How to create a Scheduled Job:
 *   1. Navigate to System Definition > Scheduled Jobs > New.
 *   2. Set Name, Run (frequency), and Time.
 *   3. Set Application and Condition.
 *   4. Paste the script in the 'Run this script' field.
 */


/* ============================================================
   JOB: Auto-close Resolved Incidents older than N days
   Frequency: Daily
   ============================================================ */
(function autoCloseResolvedIncidents() {
    var DAYS_TO_AUTO_CLOSE = 5; /* CONFIGURE */
    var cutoff = new GlideDateTime();
    cutoff.addDaysUTC(-DAYS_TO_AUTO_CLOSE);

    var gr = new GlideRecord('incident');
    gr.addQuery('state', '6');                           // Resolved
    gr.addQuery('resolved_at', '<', cutoff.getValue()); // resolved more than N days ago
    gr.query();

    var count = 0;
    while (gr.next()) {
        gr.setValue('state',     '7'); // Closed
        gr.setValue('closed_at', new GlideDateTime().getValue());
        gr.setValue('closed_by', gs.getUserID());
        gr.setValue('close_notes',
            'Auto-closed by scheduled job after ' + DAYS_TO_AUTO_CLOSE + ' days in Resolved state.');
        gr.update();
        count++;
    }
    gs.log('Auto-close job: closed ' + count + ' incidents.');
})();


/* ============================================================
   JOB: Send daily open-incident summary email to managers
   Frequency: Daily at 08:00
   ============================================================ */
(function dailyIncidentSummary() {
    var MANAGER_EMAIL = gs.getProperty('u_helpdesk_manager_email'); /* CONFIGURE */
    if (!MANAGER_EMAIL) { gs.warn('dailyIncidentSummary: no manager email configured.'); return; }

    /* Count by priority */
    var priorities = { '1': 'Critical', '2': 'High', '3': 'Moderate', '4': 'Low', '5': 'Planning' };
    var counts     = {};
    var ga = new GlideAggregate('incident');
    ga.addQuery('state', 'NOT IN', '6,7'); // open
    ga.addAggregate('COUNT');
    ga.groupBy('priority');
    ga.query();
    while (ga.next()) {
        counts[ga.getValue('priority')] = ga.getAggregate('COUNT');
    }

    var html = '<h2>Daily Open Incident Summary — ' + new GlideDateTime().getDisplayValue() + '</h2>';
    html += '<table border="1" cellpadding="5"><tr><th>Priority</th><th>Open Count</th></tr>';
    for (var p in priorities) {
        html += '<tr><td>' + priorities[p] + '</td><td>' + (counts[p] || 0) + '</td></tr>';
    }
    html += '</table>';

    var email = new GlideEmailOutbound();
    email.setTo(MANAGER_EMAIL);
    email.setSubject('Daily Incident Summary — ' + new GlideDate().getDisplayValue());
    email.setBody(html);
    email.send();
})();


/* ============================================================
   JOB: Purge old audit log records
   Frequency: Weekly
   ============================================================ */
(function purgeAuditLogs() {
    var KEEP_DAYS = 90; /* CONFIGURE: retain audit logs for this many days */
    var cutoff = new GlideDateTime();
    cutoff.addDaysUTC(-KEEP_DAYS);

    var gr = new GlideRecord('u_incident_audit_log'); /* CONFIGURE: your audit table */
    gr.addQuery('u_changed_at', '<', cutoff.getValue());
    var count = gr.deleteMultiple();
    gs.log('Purge audit logs: deleted entries older than ' + KEEP_DAYS + ' days.');
})();


/* ============================================================
   JOB: Sync user data from external directory (stub)
   Frequency: Every 4 hours
   ============================================================ */
(function syncUserDirectory() {
    var rm = new sn_ws.RESTMessageV2();
    rm.setEndpoint(gs.getProperty('u_directory_api_url')); /* CONFIGURE */
    rm.setHttpMethod('GET');
    rm.setRequestHeader('Authorization', 'Bearer ' + gs.getProperty('u_directory_api_token'));
    rm.setHttpTimeout(30000);

    try {
        var response = rm.execute();
        if (response.getStatusCode() !== 200) {
            gs.error('syncUserDirectory: API returned ' + response.getStatusCode());
            return;
        }
        var users = JSON.parse(response.getBody()).users; /* CONFIGURE: adjust path */
        var synced = 0;
        for (var i = 0; i < users.length; i++) {
            var u = users[i];
            var gr = new GlideRecord('sys_user');
            gr.addQuery('email', u.email);
            gr.setLimit(1);
            gr.query();
            if (gr.next()) {
                gr.setValue('u_external_id', u.id); /* CONFIGURE */
                gr.setValue('department',    u.department || '');
                gr.update();
                synced++;
            }
        }
        gs.log('syncUserDirectory: synced ' + synced + ' users.');
    } catch (ex) {
        gs.error('syncUserDirectory exception: ' + ex.message);
    }
})();
