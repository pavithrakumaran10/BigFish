/**
 * Flow Designer Subflow Script Examples
 * Category: Flow Designer > Subflows
 * Description:
 *   Script Step contents for common reusable subflows.
 *   A subflow encapsulates a sequence of steps called from multiple flows.
 *
 * Subflow best practices:
 *   - Define clear Inputs and Outputs on the subflow definition.
 *   - Keep subflows focused on a single logical operation.
 *   - Return meaningful error indicators as Outputs instead of throwing.
 */


/* ============================================================
   SUBFLOW: Escalate Incident
   Purpose:  Bump priority, reassign to escalation group, and notify manager.
   Inputs:   incident_sys_id (String), new_priority (String), reason (String)
   Outputs:  success (Boolean), updated_group (String)
   ============================================================ */
(function execute(inputs, outputs) {
    var sysId       = inputs.incident_sys_id;
    var newPriority = inputs.new_priority || '1';
    var reason      = inputs.reason       || 'Escalated via Flow';

    var gr = new GlideRecord('incident');
    if (!gr.get(sysId)) {
        outputs.success       = false;
        outputs.updated_group = '';
        return;
    }

    /* CONFIGURE: sys_id of your escalation assignment group */
    var escalationGroup = gs.getProperty('u_escalation_group_sys_id');

    gr.setValue('priority',         newPriority);
    gr.setValue('assignment_group', escalationGroup);
    gr.setValue('work_notes', reason + ' — Escalated by Flow Designer on ' +
        new GlideDateTime().getDisplayValue());
    gr.update();

    /* Notify the manager */
    var managerGr = new GlideRecord('sys_user_group');
    if (managerGr.get(escalationGroup) && !managerGr.manager.nil()) {
        gs.eventQueue('incident.escalated', gr,  /* CONFIGURE: register this event */
            managerGr.manager.getDisplayValue(),
            gr.getValue('number'));
    }

    outputs.success       = true;
    outputs.updated_group = escalationGroup;
})(inputs, outputs);


/* ============================================================
   SUBFLOW: Provision User Access
   Purpose:  Add a user to a group and assign a role.
   Inputs:   user_sys_id (String), group_sys_id (String), role_name (String)
   Outputs:  success (Boolean), error (String)
   ============================================================ */
(function execute(inputs, outputs) {
    var userSysId  = inputs.user_sys_id;
    var groupSysId = inputs.group_sys_id;
    var roleName   = inputs.role_name;

    /* Add user to group */
    var existing = new GlideRecord('sys_user_grmember');
    existing.addQuery('user',  userSysId);
    existing.addQuery('group', groupSysId);
    existing.setLimit(1);
    existing.query();
    if (!existing.next()) {
        var membership = new GlideRecord('sys_user_grmember');
        membership.initialize();
        membership.setValue('user',  userSysId);
        membership.setValue('group', groupSysId);
        membership.insert();
    }

    /* Assign role */
    if (roleName) {
        var role = new GlideRecord('sys_user_role');
        role.addQuery('name', roleName);
        role.setLimit(1);
        role.query();
        if (role.next()) {
            var userRole = new GlideRecord('sys_user_has_role');
            userRole.addQuery('user',  userSysId);
            userRole.addQuery('role',  role.getUniqueValue());
            userRole.setLimit(1);
            userRole.query();
            if (!userRole.next()) {
                var newRole = new GlideRecord('sys_user_has_role');
                newRole.initialize();
                newRole.setValue('user',  userSysId);
                newRole.setValue('role',  role.getUniqueValue());
                newRole.setValue('state', 'active');
                newRole.insert();
            }
        } else {
            outputs.success = false;
            outputs.error   = 'Role not found: ' + roleName;
            return;
        }
    }

    outputs.success = true;
    outputs.error   = '';
})(inputs, outputs);


/* ============================================================
   SUBFLOW: Archive Old Records
   Purpose:  Copy records older than X days to an archive table, then delete originals.
   Inputs:   source_table (String), archive_table (String), days_old (Integer),
             fields (String, comma-separated)
   Outputs:  archived_count (Integer), success (Boolean)
   ============================================================ */
(function execute(inputs, outputs) {
    var sourceTable  = inputs.source_table;  /* CONFIGURE */
    var archiveTable = inputs.archive_table; /* CONFIGURE */
    var daysOld      = parseInt(inputs.days_old || '90', 10);
    var fieldList    = (inputs.fields || 'short_description,sys_created_on').split(',');

    var cutoff = new GlideDateTime();
    cutoff.addDaysUTC(-daysOld);

    var gr = new GlideRecord(sourceTable);
    gr.addQuery('sys_created_on', '<', cutoff.getValue());
    gr.query();

    var count = 0;
    while (gr.next()) {
        /* Copy to archive table */
        var archive = new GlideRecord(archiveTable);
        archive.initialize();
        archive.setValue('u_original_sys_id', gr.getUniqueValue());
        for (var i = 0; i < fieldList.length; i++) {
            var f = fieldList[i].trim();
            if (f) archive.setValue(f, gr.getValue(f));
        }
        archive.insert();

        /* Delete the original */
        gr.deleteRecord();
        count++;
    }

    outputs.archived_count = count;
    outputs.success        = true;
})(inputs, outputs);
