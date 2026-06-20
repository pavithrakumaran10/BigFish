/**
 * Script Include: GlideRecordUtils
 * Category:       GlideRecord
 * Description:    Safe, reusable CRUD wrappers and query helpers for GlideRecord.
 * Client Callable: false
 * Scope:          Global
 *
 * Usage:
 *   var gru = new GlideRecordUtils();
 *   var rec  = gru.getById('incident', 'sys_id_here');
 *   var list = gru.query('incident', { active: true, state: 1 }, 10);
 *   var sys  = gru.insert('incident', { short_description: 'Test', urgency: 2 });
 */
var GlideRecordUtils = Class.create();
GlideRecordUtils.prototype = {
    initialize: function() {},

    /**
     * Get a single record by sys_id.
     * @param {string} table
     * @param {string} sysId
     * @returns {GlideRecord|null}
     */
    getById: function(table, sysId) {
        var gr = new GlideRecord(table);
        if (gr.get(sysId)) return gr;
        return null;
    },

    /**
     * Get a single record by a specific field value.
     * @param {string} table
     * @param {string} field
     * @param {string} value
     * @returns {GlideRecord|null}
     */
    getByField: function(table, field, value) {
        var gr = new GlideRecord(table);
        gr.addQuery(field, value);
        gr.setLimit(1);
        gr.query();
        if (gr.next()) return gr;
        return null;
    },

    /**
     * Query a table with a field/value map and optional limit.
     * @param {string} table
     * @param {Object} conditions  - { field: value, ... }
     * @param {number} [limit]     - max records to return
     * @returns {GlideRecord} - positioned before first result; caller must call .next()
     */
    query: function(table, conditions, limit) {
        var gr = new GlideRecord(table);
        if (conditions) {
            for (var field in conditions) {
                if (conditions.hasOwnProperty(field)) {
                    gr.addQuery(field, conditions[field]);
                }
            }
        }
        if (limit) gr.setLimit(limit);
        gr.query();
        return gr;
    },

    /**
     * Insert a new record and return its sys_id.
     * @param {string} table
     * @param {Object} fields  - { field: value, ... }
     * @returns {string|null} sys_id of the new record, or null on failure
     */
    insert: function(table, fields) {
        var gr = new GlideRecord(table);
        gr.initialize();
        for (var field in fields) {
            if (fields.hasOwnProperty(field)) {
                gr.setValue(field, fields[field]);
            }
        }
        var sysId = gr.insert();
        if (!sysId) {
            gs.error('GlideRecordUtils.insert: failed to insert into ' + table);
            return null;
        }
        return sysId;
    },

    /**
     * Update a record by sys_id.
     * @param {string} table
     * @param {string} sysId
     * @param {Object} fields  - { field: value, ... }
     * @returns {boolean}
     */
    updateById: function(table, sysId, fields) {
        var gr = this.getById(table, sysId);
        if (!gr) {
            gs.warn('GlideRecordUtils.updateById: record not found — ' + table + ' / ' + sysId);
            return false;
        }
        for (var field in fields) {
            if (fields.hasOwnProperty(field)) {
                gr.setValue(field, fields[field]);
            }
        }
        gr.update();
        return true;
    },

    /**
     * Soft-delete (set active=false) a record by sys_id.
     * @param {string} table
     * @param {string} sysId
     * @returns {boolean}
     */
    deactivate: function(table, sysId) {
        return this.updateById(table, sysId, { active: false });
    },

    /**
     * Hard-delete a record by sys_id.
     * Use with caution — prefer deactivate() in most cases.
     * @param {string} table
     * @param {string} sysId
     * @returns {boolean}
     */
    deleteById: function(table, sysId) {
        var gr = this.getById(table, sysId);
        if (!gr) return false;
        gr.deleteRecord();
        return true;
    },

    /**
     * Return the count of records matching conditions.
     * @param {string} table
     * @param {Object} conditions
     * @returns {number}
     */
    count: function(table, conditions) {
        var ga = new GlideAggregate(table);
        if (conditions) {
            for (var field in conditions) {
                if (conditions.hasOwnProperty(field)) {
                    ga.addQuery(field, conditions[field]);
                }
            }
        }
        ga.addAggregate('COUNT');
        ga.query();
        if (ga.next()) return parseInt(ga.getAggregate('COUNT'), 10);
        return 0;
    },

    /**
     * Check whether a record with the given sys_id exists.
     * @param {string} table
     * @param {string} sysId
     * @returns {boolean}
     */
    exists: function(table, sysId) {
        return this.getById(table, sysId) !== null;
    },

    /**
     * Return an array of objects from a query.
     * @param {string} table
     * @param {Object} conditions
     * @param {string[]} fields   - list of field names to include
     * @param {number}  [limit]
     * @returns {Object[]}
     */
    toArray: function(table, conditions, fields, limit) {
        var result = [];
        var gr = this.query(table, conditions, limit);
        while (gr.next()) {
            var row = { sys_id: gr.getUniqueValue() };
            for (var i = 0; i < fields.length; i++) {
                row[fields[i]] = gr.getValue(fields[i]);
            }
            result.push(row);
        }
        return result;
    },

    type: 'GlideRecordUtils'
};
