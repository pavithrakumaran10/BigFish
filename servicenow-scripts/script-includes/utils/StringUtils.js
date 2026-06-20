/**
 * Script Include: StringUtils
 * Category:       Utilities > String
 * Description:    Reusable string manipulation methods for ServiceNow development.
 * Client Callable: false  (set to true and add GlideAjax wrappers if needed)
 * Scope:          Global
 *
 * Usage:
 *   var su = new StringUtils();
 *   var camel = su.toCamelCase('hello world');   // 'helloWorld'
 *   var snake = su.toSnakeCase('HelloWorld');      // 'hello_world'
 *   var safe  = su.truncate('Long string...', 20);
 */
var StringUtils = Class.create();
StringUtils.prototype = {
    initialize: function() {},

    /** Remove leading and trailing whitespace. */
    trim: function(str) {
        if (!str) return '';
        return String(str).replace(/^\s+|\s+$/g, '');
    },

    /** Convert to camelCase. e.g. 'hello world foo' → 'helloWorldFoo' */
    toCamelCase: function(str) {
        if (!str) return '';
        return String(str)
            .toLowerCase()
            .replace(/[^a-zA-Z0-9]+(.)/g, function(match, chr) {
                return chr.toUpperCase();
            });
    },

    /** Convert to snake_case. e.g. 'HelloWorld' → 'hello_world' */
    toSnakeCase: function(str) {
        if (!str) return '';
        return String(str)
            .replace(/([A-Z])/g, function(m) { return '_' + m.toLowerCase(); })
            .replace(/[\s-]+/g, '_')
            .replace(/^_/, '')
            .toLowerCase();
    },

    /** Truncate to maxLength chars, appending ellipsis if truncated. */
    truncate: function(str, maxLength) {
        if (!str) return '';
        str = String(str);
        if (str.length <= maxLength) return str;
        return str.substring(0, maxLength - 3) + '...';
    },

    /** Return true if the string is null, undefined, or only whitespace. */
    isEmpty: function(str) {
        return !str || String(str).trim() === '';
    },

    /** Case-insensitive substring check. */
    containsIgnoreCase: function(str, searchStr) {
        if (!str || !searchStr) return false;
        return String(str).toLowerCase().indexOf(String(searchStr).toLowerCase()) !== -1;
    },

    /** Left-pad a string to a target length. e.g. padLeft('5', 3, '0') → '005' */
    padLeft: function(str, length, padChar) {
        str = String(str || '');
        padChar = padChar || ' ';
        while (str.length < length) str = padChar + str;
        return str;
    },

    /** Replace all occurrences of find with replace. */
    replaceAll: function(str, find, replace) {
        if (!str) return '';
        return String(str).split(find).join(replace);
    },

    /** Capitalize the first letter. */
    capitalize: function(str) {
        if (!str) return '';
        str = String(str);
        return str.charAt(0).toUpperCase() + str.slice(1);
    },

    /** Return an array of all regex matches in str. */
    extractMatches: function(str, pattern) {
        if (!str) return [];
        var regex = new RegExp(pattern, 'g');
        var matches = [];
        var match;
        while ((match = regex.exec(str)) !== null) {
            matches.push(match[0]);
        }
        return matches;
    },

    /** Escape HTML special characters to prevent XSS. */
    escapeHTML: function(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    },

    /** Convert a comma-separated string to an array, trimming each item. */
    csvToArray: function(csv) {
        if (!csv) return [];
        return String(csv).split(',').map(function(item) { return item.trim(); }).filter(Boolean);
    },

    /** Convert an array to a comma-separated string. */
    arrayToCSV: function(arr) {
        if (!arr || !arr.length) return '';
        return arr.join(',');
    },

    type: 'StringUtils'
};
