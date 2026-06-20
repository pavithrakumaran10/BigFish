/**
 * Flow Designer Custom Action Scripts
 * Category: Flow Designer > Custom Actions
 * Description: Reusable action script steps for common Flow Designer tasks.
 *
 * How to create a Custom Action:
 *   1. Navigate to Process Automation > Flow Designer > New > Action.
 *   2. Define Inputs and Outputs in the respective tabs.
 *   3. Add a 'Script Step' and paste the relevant script below.
 *   4. Publish the action to make it available in flows.
 */


/* ============================================================
   ACTION: Get Record Fields
   Inputs:  table_name (String), record_sys_id (String), field_names (String, comma-separated)
   Outputs: field_values (String, JSON), found (Boolean)
   ============================================================ */
(function execute(inputs, outputs) {
    var table   = inputs.table_name;    /* CONFIGURE */
    var sysId   = inputs.record_sys_id; /* CONFIGURE */
    var fields  = (inputs.field_names || '').split(',');

    var gr = new GlideRecord(table);
    if (!gr.get(sysId)) {
        outputs.found        = false;
        outputs.field_values = '{}';
        return;
    }

    var result = { sys_id: gr.getUniqueValue() };
    for (var i = 0; i < fields.length; i++) {
        var f = fields[i].trim();
        if (f) result[f] = gr.getValue(f);
    }

    outputs.found        = true;
    outputs.field_values = JSON.stringify(result);
})(inputs, outputs);


/* ============================================================
   ACTION: Update Record Fields
   Inputs:  table_name (String), record_sys_id (String),
            field_updates (String, JSON key-value pairs)
   Outputs: success (Boolean), error (String)
   ============================================================ */
(function execute(inputs, outputs) {
    var table   = inputs.table_name;
    var sysId   = inputs.record_sys_id;
    var updates;

    try {
        updates = JSON.parse(inputs.field_updates || '{}');
    } catch (e) {
        outputs.success = false;
        outputs.error   = 'Invalid JSON in field_updates: ' + e.message;
        return;
    }

    var gr = new GlideRecord(table);
    if (!gr.get(sysId)) {
        outputs.success = false;
        outputs.error   = 'Record not found: ' + table + ' / ' + sysId;
        return;
    }

    for (var field in updates) {
        if (updates.hasOwnProperty(field)) {
            gr.setValue(field, updates[field]);
        }
    }
    gr.update();
    outputs.success = true;
    outputs.error   = '';
})(inputs, outputs);


/* ============================================================
   ACTION: Call External REST API
   Inputs:  method (String), url (String), body_json (String), bearer_token (String)
   Outputs: status_code (Integer), response_body (String), success (Boolean)
   ============================================================ */
(function execute(inputs, outputs) {
    var method = (inputs.method || 'GET').toUpperCase();
    var url    = inputs.url;
    var body   = inputs.body_json   || null;
    var token  = inputs.bearer_token || '';

    try {
        var rm = new sn_ws.RESTMessageV2();
        rm.setEndpoint(url);
        rm.setHttpMethod(method);
        rm.setRequestHeader('Content-Type', 'application/json');
        rm.setRequestHeader('Accept',       'application/json');
        if (token) rm.setRequestHeader('Authorization', 'Bearer ' + token);
        if (body && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
            rm.setRequestBody(body);
        }
        rm.setHttpTimeout(20000);

        var response = rm.execute();
        outputs.status_code    = response.getStatusCode();
        outputs.response_body  = response.getBody();
        outputs.success        = response.getStatusCode() >= 200 && response.getStatusCode() < 300;
    } catch (ex) {
        outputs.status_code   = -1;
        outputs.response_body = ex.message;
        outputs.success       = false;
    }
})(inputs, outputs);


/* ============================================================
   ACTION: Send Email Notification
   Inputs:  to_email (String), subject (String), body_html (String),
            cc_email (String)
   Outputs: success (Boolean)
   ============================================================ */
(function execute(inputs, outputs) {
    try {
        var email = new GlideEmailOutbound();
        email.setTo(inputs.to_email);
        email.setSubject(inputs.subject);
        email.setBody(inputs.body_html);
        if (inputs.cc_email) email.setCc(inputs.cc_email);
        email.send();
        outputs.success = true;
    } catch (ex) {
        gs.error('Flow Action SendEmail failed: ' + ex.message);
        outputs.success = false;
    }
})(inputs, outputs);


/* ============================================================
   ACTION: Create Child Task
   Inputs:  parent_table (String), parent_sys_id (String),
            task_table (String), short_description (String),
            assignment_group (String), due_date (String)
   Outputs: task_sys_id (String), task_number (String), success (Boolean)
   ============================================================ */
(function execute(inputs, outputs) {
    try {
        var gr = new GlideRecord(inputs.task_table || 'task'); /* CONFIGURE */
        gr.initialize();
        gr.setValue('short_description', inputs.short_description);
        gr.setValue('parent',            inputs.parent_sys_id);
        if (inputs.assignment_group) gr.setValue('assignment_group', inputs.assignment_group);
        if (inputs.due_date)         gr.setValue('due_date', inputs.due_date);

        var sysId = gr.insert();
        if (!sysId) throw new Error('insert() returned null');

        var created = new GlideRecord(inputs.task_table || 'task');
        created.get(sysId);

        outputs.task_sys_id  = sysId;
        outputs.task_number  = created.getValue('number');
        outputs.success      = true;
    } catch (ex) {
        outputs.task_sys_id = '';
        outputs.task_number = '';
        outputs.success     = false;
        gs.error('Flow Action CreateChildTask failed: ' + ex.message);
    }
})(inputs, outputs);
