/**
 * Script Include: UserGroupUtils
 * Category:       User & Group
 * Description:    Helper methods for users, roles, and groups in ServiceNow.
 * Client Callable: true  (GlideAjax-compatible methods marked with *)
 * Scope:          Global
 *
 * Usage:
 *   var ugu = new UserGroupUtils();
 *   var mgr = ugu.getManager('group_sys_id');
 *   var has = ugu.userHasRole(gs.getUserID(), 'itil');
 *   var mem = ugu.getGroupMembers('group_sys_id');  // returns array of sys_ids
 */
var UserGroupUtils = Class.create();
UserGroupUtils.prototype = Object.extendsObject(AbstractAjaxProcessor, {
    initialize: function() {},

    /** Return the sys_user GlideRecord for a given sys_id. */
    getUserById: function(userSysId) {
        var gr = new GlideRecord('sys_user');
        if (gr.get(userSysId)) return gr;
        return null;
    },

    /** Return the sys_user GlideRecord for a given username (user_name field). */
    getUserByUsername: function(username) {
        var gr = new GlideRecord('sys_user');
        gr.addQuery('user_name', username);
        gr.setLimit(1);
        gr.query();
        if (gr.next()) return gr;
        return null;
    },

    /** Return the current logged-in user's sys_id. */
    getCurrentUserId: function() {
        return gs.getUserID();
    },

    /** Return the current user's full name. */
    getCurrentUserName: function() {
        return gs.getUserDisplayName();
    },

    /**
     * Check whether a user has a specific role.
     * @param {string} userSysId
     * @param {string} roleName   - e.g. 'itil', 'admin'
     * @returns {boolean}
     */
    userHasRole: function(userSysId, roleName) {
        var gr = new GlideRecord('sys_user_has_role');
        gr.addQuery('user', userSysId);
        gr.addQuery('role.name', roleName);
        gr.addQuery('state', 'active');
        gr.setLimit(1);
        gr.query();
        return gr.next();
    },

    /**
     * Check whether the current user has a role.
     * @param {string} roleName
     * @returns {boolean}
     */
    currentUserHasRole: function(roleName) {
        return gs.hasRole(roleName);
    },

    /**
     * Return sys_ids of all members of a group.
     * @param {string} groupSysId
     * @returns {string[]}
     */
    getGroupMembers: function(groupSysId) {
        var members = [];
        var gr = new GlideRecord('sys_user_grmember');
        gr.addQuery('group', groupSysId);
        gr.query();
        while (gr.next()) {
            members.push(gr.getValue('user'));
        }
        return members;
    },

    /**
     * Return the sys_id of the group manager.
     * @param {string} groupSysId
     * @returns {string|null}
     */
    getManager: function(groupSysId) {
        var gr = new GlideRecord('sys_user_group');
        if (gr.get(groupSysId)) return gr.getValue('manager');
        return null;
    },

    /**
     * Return all groups a user belongs to.
     * @param {string} userSysId
     * @returns {string[]} array of group sys_ids
     */
    getUserGroups: function(userSysId) {
        var groups = [];
        var gr = new GlideRecord('sys_user_grmember');
        gr.addQuery('user', userSysId);
        gr.query();
        while (gr.next()) {
            groups.push(gr.getValue('group'));
        }
        return groups;
    },

    /**
     * Check whether a user is in a specific group.
     * @param {string} userSysId
     * @param {string} groupSysId
     * @returns {boolean}
     */
    isUserInGroup: function(userSysId, groupSysId) {
        var gr = new GlideRecord('sys_user_grmember');
        gr.addQuery('user', userSysId);
        gr.addQuery('group', groupSysId);
        gr.setLimit(1);
        gr.query();
        return gr.next();
    },

    /**
     * GlideAjax wrapper: check if the current user is in a group.
     * Call from client with GlideAjax, sysparm_name = 'ajaxIsUserInGroup'.
     */
    ajaxIsUserInGroup: function() {
        var groupSysId = this.getParameter('sysparm_group_id');
        return String(this.isUserInGroup(gs.getUserID(), groupSysId));
    },

    type: 'UserGroupUtils'
});
