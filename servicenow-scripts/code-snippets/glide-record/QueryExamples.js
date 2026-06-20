/**
 * Snippet Collection: Advanced GlideRecord Query Patterns
 * Context: Business Rules, Script Includes
 * Description: Common query patterns beyond basic addQuery.
 */

/* ---- Date range queries ---- */
var gr = new GlideRecord('incident');
var start = new GlideDateTime();
start.addDaysUTC(-7); // last 7 days
gr.addQuery('sys_created_on', '>=', start.getValue());
gr.addQuery('sys_created_on', '<=', new GlideDateTime().getValue());
gr.query();

/* ---- Query on a reference field's property ---- */
// Find incidents where the caller's department = 'IT'
var gr2 = new GlideRecord('incident');
gr2.addQuery('caller_id.department.name', 'Information Technology');
gr2.query();

/* ---- Dot-walk to related fields ---- */
var gr3 = new GlideRecord('incident');
gr3.addQuery('assignment_group.name', 'Service Desk');
gr3.addQuery('assigned_to.active',    true);
gr3.query();
while (gr3.next()) {
    var assigneeName = gr3.getDisplayValue('assigned_to'); // display value of reference
    var groupSysId   = gr3.getValue('assignment_group');   // raw sys_id
}

/* ---- NULL / NOT NULL checks ---- */
var gr4 = new GlideRecord('incident');
gr4.addQuery('assigned_to', 'ISEMPTY', '');    // unassigned
gr4.addQuery('resolved_at', 'ISNOTEMPTY', ''); // has a resolved_at date
gr4.query();

/* ---- IN / NOT IN list ---- */
var gr5 = new GlideRecord('incident');
gr5.addQuery('state', 'IN',     '1,2,3');   // New, In Progress, On Hold
gr5.addQuery('state', 'NOT IN', '6,7');     // not Resolved, not Closed
gr5.query();

/* ---- CONTAINS / STARTSWITH / ENDSWITH ---- */
var gr6 = new GlideRecord('incident');
gr6.addQuery('short_description', 'CONTAINS', 'network');
gr6.query();

/* ---- Query with active join (EXISTS) ---- */
// Incidents that have at least one active child task
var gr7   = new GlideRecord('incident');
var join  = gr7.addJoinQuery('task', 'sys_id', 'parent');
join.addCondition('state', 'NOT IN', '3,4'); // active child tasks
gr7.query();

/* ---- Paginated query (process in batches of 100) ---- */
var BATCH_SIZE = 100;
var offset     = 0;
var hasMore    = true;
while (hasMore) {
    var page = new GlideRecord('incident');
    page.addQuery('active', true);
    page.chooseWindow(offset, offset + BATCH_SIZE);
    page.query();
    var count = 0;
    while (page.next()) {
        count++;
        // process record...
    }
    hasMore = (count === BATCH_SIZE);
    offset += BATCH_SIZE;
}

/* ---- Get field display value vs internal value ---- */
var gr8 = new GlideRecord('incident');
gr8.setLimit(1);
gr8.query();
if (gr8.next()) {
    var stateInternal = gr8.getValue('state');          // '1'
    var stateDisplay  = gr8.getDisplayValue('state');   // 'New'
    var callerSysId   = gr8.getValue('caller_id');      // sys_id
    var callerName    = gr8.getDisplayValue('caller_id'); // 'John Smith'
}
