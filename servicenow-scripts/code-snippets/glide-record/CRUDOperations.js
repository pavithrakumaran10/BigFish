/**
 * Snippet Collection: GlideRecord CRUD Operations
 * Context: Business Rules, Script Includes, Background Scripts
 * Description: Complete set of Create, Read, Update, Delete patterns.
 */

/* ============================================================
   CREATE - Insert a new record
   ============================================================ */

// Basic insert
var gr = new GlideRecord('incident');
gr.initialize();
gr.setValue('short_description', 'Network outage in Building A');
gr.setValue('category',          'network');
gr.setValue('urgency',           '1');
gr.setValue('impact',            '1');
var newSysId = gr.insert();
gs.log('Created incident: ' + newSysId);

// Insert with a reference field (set by sys_id)
var gr2 = new GlideRecord('incident');
gr2.initialize();
gr2.setValue('short_description', 'Printer not working');
gr2.setValue('caller_id',         '6816f79cc0a8016401c5a33be04be441'); // sys_id of user
gr2.insert();

// Insert and immediately retrieve the created record
var gr3 = new GlideRecord('sc_task');
gr3.initialize();
gr3.setValue('short_description', 'Provision laptop');
var taskId = gr3.insert();
if (taskId) {
    var created = new GlideRecord('sc_task');
    created.get(taskId);
    gs.log('Task number: ' + created.getValue('number'));
}


/* ============================================================
   READ - Query records
   ============================================================ */

// Get single record by sys_id
var inc = new GlideRecord('incident');
if (inc.get('abc123def456abc123def456abc123de')) {
    gs.log('Found: ' + inc.getValue('short_description'));
}

// Get single record by field value
var inc2 = new GlideRecord('incident');
if (inc2.get('number', 'INC0010001')) {
    gs.log('Priority: ' + inc2.getValue('priority'));
}

// Query multiple records with conditions
var gr4 = new GlideRecord('incident');
gr4.addQuery('state', '1');            // State = New
gr4.addQuery('priority', '1');         // Priority = 1 - Critical
gr4.orderByDesc('sys_created_on');     // Newest first
gr4.setLimit(50);                      // Max 50 records
gr4.query();
while (gr4.next()) {
    gs.log(gr4.getValue('number') + ': ' + gr4.getValue('short_description'));
}

// Query with OR condition
var gr5 = new GlideRecord('incident');
var qc   = gr5.addQuery('state', '1');  // state = New
qc.addOrCondition('state', '2');        // OR state = In Progress
gr5.query();
while (gr5.next()) { /* ... */ }

// Query with encoded query string
var gr6 = new GlideRecord('incident');
gr6.addEncodedQuery('active=true^priority=1^ORpriority=2^stateNOT IN6,7');
gr6.query();
while (gr6.next()) { /* ... */ }

// GlideAggregate: count records
var ga = new GlideAggregate('incident');
ga.addQuery('state', 'NOT IN', '6,7');
ga.addAggregate('COUNT');
ga.query();
if (ga.next()) {
    gs.log('Open incidents: ' + ga.getAggregate('COUNT'));
}

// GlideAggregate: group by field
var ga2 = new GlideAggregate('incident');
ga2.addQuery('active', true);
ga2.addAggregate('COUNT');
ga2.groupBy('category');
ga2.query();
while (ga2.next()) {
    gs.log(ga2.getValue('category') + ': ' + ga2.getAggregate('COUNT'));
}


/* ============================================================
   UPDATE - Modify existing records
   ============================================================ */

// Update single record by sys_id
var upd = new GlideRecord('incident');
if (upd.get('abc123def456abc123def456abc123de')) {
    upd.setValue('state',      '2');
    upd.setValue('work_notes', 'Assigned to Level 2 support.');
    upd.update();
}

// Update all records matching a condition (bulk update)
var bulk = new GlideRecord('incident');
bulk.addQuery('assignment_group', 'old_group_sys_id');
bulk.addQuery('state', 'NOT IN', '6,7');
bulk.query();
while (bulk.next()) {
    bulk.setValue('assignment_group', 'new_group_sys_id');
    bulk.update();
}

// Update with autoSysFields=false (don't change sys_updated_on)
var upd2 = new GlideRecord('incident');
if (upd2.get('sys_id_here')) {
    upd2.autoSysFields(false);
    upd2.setValue('u_background_field', 'silent_update');
    upd2.update();
}


/* ============================================================
   DELETE - Remove records
   ============================================================ */

// Delete single record
var del1 = new GlideRecord('u_temp_staging');
if (del1.get('sys_id_here')) {
    del1.deleteRecord();
}

// Delete all records matching a condition
var del2 = new GlideRecord('u_temp_staging');
del2.addQuery('u_processed', true);
del2.deleteMultiple(); // WARNING: no per-record hooks or watermarks
