/**
 * Snippet Collection: Inbound REST API (Scripted REST API)
 * Context: System Web Services > Scripted REST APIs
 * Description: Build custom REST endpoints exposed by your ServiceNow instance.
 *
 * How to set up:
 *   1. Navigate to System Web Services > Scripted REST APIs > New.
 *   2. Set Name, API ID, and namespace.
 *   3. Add Resources (GET, POST, PUT, DELETE) with the scripts below.
 */

/* ---- GET /api/{namespace}/incidents/{number} ---- */
(function process(request, response) {
    var number = request.pathParams.number; /* CONFIGURE: path parameter name */

    if (!number) {
        response.setStatus(400);
        response.setBody({ error: 'Incident number is required.' });
        return;
    }

    var gr = new GlideRecord('incident');
    gr.addQuery('number', number);
    gr.setLimit(1);
    gr.query();

    if (!gr.next()) {
        response.setStatus(404);
        response.setBody({ error: 'Incident ' + number + ' not found.' });
        return;
    }

    response.setStatus(200);
    response.setBody({
        sys_id:            gr.getUniqueValue(),
        number:            gr.getValue('number'),
        short_description: gr.getValue('short_description'),
        state:             gr.getDisplayValue('state'),
        priority:          gr.getDisplayValue('priority'),
        assigned_to:       gr.getDisplayValue('assigned_to'),
        created_on:        gr.getValue('sys_created_on')
    });
})(request, response);


/* ---- POST /api/{namespace}/incidents ---- */
(function process(request, response) {
    var body = request.body ? request.body.data : null;

    if (!body || !body.short_description) {
        response.setStatus(400);
        response.setBody({ error: 'short_description is required.' });
        return;
    }

    /* Validate caller exists if provided */
    if (body.caller_id) {
        var user = new GlideRecord('sys_user');
        if (!user.get(body.caller_id)) {
            response.setStatus(400);
            response.setBody({ error: 'Caller with sys_id ' + body.caller_id + ' not found.' });
            return;
        }
    }

    var gr = new GlideRecord('incident');
    gr.initialize();
    gr.setValue('short_description', body.short_description);
    gr.setValue('description',       body.description       || '');
    gr.setValue('category',          body.category          || 'inquiry');
    gr.setValue('urgency',           body.urgency           || '3');
    gr.setValue('impact',            body.impact            || '3');
    if (body.caller_id) gr.setValue('caller_id', body.caller_id);

    var sysId = gr.insert();
    if (!sysId) {
        response.setStatus(500);
        response.setBody({ error: 'Failed to create incident.' });
        return;
    }

    /* Fetch the created record for the response */
    var created = new GlideRecord('incident');
    created.get(sysId);
    response.setStatus(201);
    response.setBody({
        sys_id:  sysId,
        number:  created.getValue('number'),
        message: 'Incident created successfully.'
    });
})(request, response);


/* ---- PATCH /api/{namespace}/incidents/{sys_id} ---- */
(function process(request, response) {
    var sysId = request.pathParams.sys_id;
    var body  = request.body ? request.body.data : null;

    if (!sysId || !body) {
        response.setStatus(400);
        response.setBody({ error: 'sys_id and request body are required.' });
        return;
    }

    var gr = new GlideRecord('incident');
    if (!gr.get(sysId)) {
        response.setStatus(404);
        response.setBody({ error: 'Incident not found.' });
        return;
    }

    /* Only update fields that are present in the request body */
    var allowedFields = ['short_description', 'description', 'state', 'work_notes',
                         'assignment_group', 'assigned_to', 'urgency', 'impact'];
    for (var i = 0; i < allowedFields.length; i++) {
        if (body[allowedFields[i]] !== undefined) {
            gr.setValue(allowedFields[i], body[allowedFields[i]]);
        }
    }
    gr.update();

    response.setStatus(200);
    response.setBody({ message: 'Incident updated.', sys_id: sysId });
})(request, response);
