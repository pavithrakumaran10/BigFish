/**
 * Script Include: NotificationUtils
 * Category:       Notification
 * Description:    Programmatic email and event helpers for ServiceNow.
 * Client Callable: false
 * Scope:          Global
 *
 * Usage:
 *   var nu = new NotificationUtils();
 *   nu.sendEmail('user@example.com', 'Subject', '<p>Body HTML</p>');
 *   nu.fireEvent('incident', current, 'incident.commented');
 */
var NotificationUtils = Class.create();
NotificationUtils.prototype = {
    initialize: function() {},

    /**
     * Send a direct email via gs.eventQueue or GlideEmailOutbound.
     * @param {string} to         - recipient email address
     * @param {string} subject
     * @param {string} body       - HTML body
     * @param {string} [from]     - sender address (defaults to instance default)
     * @param {string} [cc]       - CC address
     */
    sendEmail: function(to, subject, body, from, cc) {
        try {
            var email = new GlideEmailOutbound();
            email.setTo(to);
            email.setSubject(subject);
            email.setBody(body);
            if (from) email.setFrom(from);
            if (cc)   email.setCc(cc);
            email.send();
        } catch (ex) {
            gs.error('NotificationUtils.sendEmail failed: ' + ex.message);
        }
    },

    /**
     * Queue a platform event to trigger an email notification.
     * @param {string}      tableName - e.g. 'incident'
     * @param {GlideRecord} record    - the current record
     * @param {string}      eventName - e.g. 'incident.assigned'
     * @param {string}      [parm1]   - optional extra parameter
     * @param {string}      [parm2]   - optional extra parameter
     */
    fireEvent: function(tableName, record, eventName, parm1, parm2) {
        try {
            gs.eventQueue(eventName, record, parm1 || '', parm2 || '');
        } catch (ex) {
            gs.error('NotificationUtils.fireEvent failed: ' + ex.message);
        }
    },

    /**
     * Send an email to all members of a group.
     * @param {string} groupSysId
     * @param {string} subject
     * @param {string} body
     */
    notifyGroup: function(groupSysId, subject, body) {
        var gr = new GlideRecord('sys_user_grmember');
        gr.addQuery('group', groupSysId);
        gr.query();
        while (gr.next()) {
            var user = new GlideRecord('sys_user');
            if (user.get(gr.getValue('user'))) {
                var email = user.getValue('email');
                if (email) this.sendEmail(email, subject, body);
            }
        }
    },

    /**
     * Send an email to a user by their sys_id.
     * @param {string} userSysId
     * @param {string} subject
     * @param {string} body
     */
    notifyUserById: function(userSysId, subject, body) {
        var user = new GlideRecord('sys_user');
        if (user.get(userSysId)) {
            var email = user.getValue('email');
            if (email) {
                this.sendEmail(email, subject, body);
            } else {
                gs.warn('NotificationUtils.notifyUserById: no email address for user ' + userSysId);
            }
        }
    },

    /**
     * Create a sys_notification_recipient record (used for in-app notifications).
     * @param {string} userSysId
     * @param {string} title
     * @param {string} content
     * @param {string} [link]   - URL to navigate to on click
     */
    createInAppNotification: function(userSysId, title, content, link) {
        try {
            var notif = new GlideRecord('sys_notification_recipient');
            notif.initialize();
            notif.setValue('recipient', userSysId);
            notif.setValue('title', title);
            notif.setValue('content', content);
            if (link) notif.setValue('action_link', link);
            notif.insert();
        } catch (ex) {
            gs.error('NotificationUtils.createInAppNotification failed: ' + ex.message);
        }
    },

    type: 'NotificationUtils'
};
