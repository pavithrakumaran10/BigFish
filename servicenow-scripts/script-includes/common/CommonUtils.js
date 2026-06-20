/**
 * Script Include: CommonUtils
 * Category:       Common / General
 * Description:    Catch-all utility methods used across multiple ServiceNow applications.
 * Client Callable: true
 * Scope:          Global
 *
 * Usage:
 *   var cu = new CommonUtils();
 *   var json = cu.toJSON({ key: 'value' });
 *   var obj  = cu.fromJSON('{"key":"value"}');
 *   cu.log('info', 'My message', 'MyScript');
 */
var CommonUtils = Class.create();
CommonUtils.prototype = Object.extendsObject(AbstractAjaxProcessor, {
    initialize: function() {},

    /* ─── JSON Helpers ──────────────────────────────────────────────── */

    /** Safely serialize an object to a JSON string. */
    toJSON: function(obj) {
        try {
            return JSON.stringify(obj);
        } catch (e) {
            gs.error('CommonUtils.toJSON failed: ' + e.message);
            return '{}';
        }
    },

    /** Safely parse a JSON string. Returns null on failure. */
    fromJSON: function(str) {
        try {
            return JSON.parse(str);
        } catch (e) {
            gs.error('CommonUtils.fromJSON failed for input: ' + str);
            return null;
        }
    },

    /* ─── Logging ───────────────────────────────────────────────────── */

    /**
     * Structured log with level, message, and source.
     * @param {string} level   - 'info' | 'warn' | 'error' | 'debug'
     * @param {string} message
     * @param {string} source  - identifying script name / function
     */
    log: function(level, message, source) {
        var prefix = '[' + (source || 'CommonUtils') + '] ';
        switch (level) {
            case 'error': gs.error(prefix + message);   break;
            case 'warn':  gs.warn(prefix + message);    break;
            case 'debug': gs.debug(prefix + message);   break;
            default:      gs.log(prefix + message);     break;
        }
    },

    /* ─── System Properties ─────────────────────────────────────────── */

    /**
     * Get a system property value with a default fallback.
     * @param {string} propName
     * @param {string} [defaultValue]
     * @returns {string}
     */
    getProperty: function(propName, defaultValue) {
        var value = gs.getProperty(propName);
        return (value !== null && value !== undefined && value !== '') ? value : (defaultValue || '');
    },

    /**
     * Set a system property.
     * @param {string} propName
     * @param {string} value
     */
    setProperty: function(propName, value) {
        gs.setProperty(propName, value, 'Updated by CommonUtils');
    },

    /* ─── Session / Instance ────────────────────────────────────────── */

    /** Return the current instance URL. */
    getInstanceURL: function() {
        return 'https://' + gs.getProperty('instance_name') + '.service-now.com';
    },

    /** Return a direct URL to a record. */
    getRecordURL: function(table, sysId) {
        return this.getInstanceURL() + '/' + table + '.do?sys_id=' + sysId;
    },

    /** Return the current user's timezone. */
    getCurrentUserTimezone: function() {
        return gs.getSession().getTimeZoneName();
    },

    /* ─── GlideAjax Interface ───────────────────────────────────────── */

    /** Ajax: get a system property value (called from client via GlideAjax). */
    ajaxGetProperty: function() {
        var propName = this.getParameter('sysparm_prop_name');
        return this.getProperty(propName);
    },

    /** Ajax: get the current instance URL. */
    ajaxGetInstanceURL: function() {
        return this.getInstanceURL();
    },

    /* ─── Miscellaneous ─────────────────────────────────────────────── */

    /**
     * Generate a random alphanumeric string of given length.
     * @param {number} length
     * @returns {string}
     */
    randomString: function(length) {
        var chars  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        var result = '';
        for (var i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    },

    /**
     * Deep clone a plain JS object (no functions or GlideRecord).
     * @param {Object} obj
     * @returns {Object}
     */
    deepClone: function(obj) {
        return this.fromJSON(this.toJSON(obj));
    },

    type: 'CommonUtils'
});
