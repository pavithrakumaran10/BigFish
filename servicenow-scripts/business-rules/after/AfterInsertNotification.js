/**
 * Business Rule: AfterInsertNotification
 * Table:         /* CONFIGURE: e.g. incident */
 * When:          after
 * Insert:        true  |  Update: false  |  Delete: false  |  Query: false
 * Order:         100
 * Description:   Send a notification email after a new record is created.
 *                Uses gs.eventQueue to trigger a registered Notification.
 *
 * Prerequisites:
 *   - Create a Notification under System Policy > Email > Notifications
 *     with the event name referenced below.
 *   - Register the event under System Policy > Events > Event Registry.
 */

(function executeRule(current, previous) {

    /* ---- Fire platform event to trigger configured Notification ---- */
    /* CONFIGURE: replace with your event name registered in Event Registry */
    gs.eventQueue('incident.created', current, current.caller_id.getDisplayValue(), current.number.toString());

    /* ---- Alternative: send a direct email without an event ---- */
    /*
    var nu = new NotificationUtils();
    var callerEmail = current.caller_id.email.toString();
    if (callerEmail) {
        nu.sendEmail(
            callerEmail,
            'Your incident ' + current.number + ' has been created',
            '<p>Hello ' + current.caller_id.getDisplayValue() + ',</p>' +
            '<p>Your incident <strong>' + current.number + '</strong> has been logged.</p>' +
            '<p>Short description: ' + current.short_description + '</p>'
        );
    }
    */

    /* ---- Notify assignment group manager ---- */
    if (!current.assignment_group.nil()) {
        var groupMgr = getGroupManager(current.assignment_group.toString());
        if (groupMgr) {
            gs.eventQueue(
                'incident.assigned_to_group', /* CONFIGURE: your event name */
                current,
                groupMgr,
                current.assignment_group.getDisplayValue()
            );
        }
    }

})(current, previous);

/** Return the sys_id of a group's manager, or null. */
function getGroupManager(groupSysId) {
    var gr = new GlideRecord('sys_user_group');
    if (gr.get(groupSysId)) {
        return gr.getValue('manager');
    }
    return null;
}
