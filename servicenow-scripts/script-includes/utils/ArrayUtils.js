/**
 * Script Include: ArrayUtils
 * Category:       Utilities > Array
 * Description:    Array manipulation helpers compatible with ServiceNow's Rhino JS engine.
 * Client Callable: false
 * Scope:          Global
 *
 * Usage:
 *   var au = new ArrayUtils();
 *   var unique = au.unique([1, 2, 2, 3]);         // [1, 2, 3]
 *   var chunks = au.chunk([1,2,3,4,5], 2);        // [[1,2],[3,4],[5]]
 *   var found  = au.findBy(arr, 'id', 'abc123');  // first element where .id === 'abc123'
 */
var ArrayUtils = Class.create();
ArrayUtils.prototype = {
    initialize: function() {},

    /** Return a new array with duplicate primitives removed. */
    unique: function(arr) {
        var seen = {};
        var result = [];
        for (var i = 0; i < arr.length; i++) {
            var val = arr[i];
            if (!seen[val]) {
                seen[val] = true;
                result.push(val);
            }
        }
        return result;
    },

    /** Return a shallow copy of arr with falsy values removed. */
    compact: function(arr) {
        var result = [];
        for (var i = 0; i < arr.length; i++) {
            if (arr[i]) result.push(arr[i]);
        }
        return result;
    },

    /** Split arr into chunks of size chunkSize. */
    chunk: function(arr, chunkSize) {
        var result = [];
        for (var i = 0; i < arr.length; i += chunkSize) {
            result.push(arr.slice(i, i + chunkSize));
        }
        return result;
    },

    /** Return the first element in arr where arr[n][key] === value. */
    findBy: function(arr, key, value) {
        for (var i = 0; i < arr.length; i++) {
            if (arr[i][key] === value) return arr[i];
        }
        return null;
    },

    /** Return all elements in arr where arr[n][key] === value. */
    filterBy: function(arr, key, value) {
        var result = [];
        for (var i = 0; i < arr.length; i++) {
            if (arr[i][key] === value) result.push(arr[i]);
        }
        return result;
    },

    /** Pluck a single property from every element. e.g. pluck(users, 'name') */
    pluck: function(arr, key) {
        var result = [];
        for (var i = 0; i < arr.length; i++) {
            result.push(arr[i][key]);
        }
        return result;
    },

    /** Return true if arr contains value. */
    contains: function(arr, value) {
        for (var i = 0; i < arr.length; i++) {
            if (arr[i] === value) return true;
        }
        return false;
    },

    /** Group array elements by a key. Returns an object { keyValue: [elements...] }. */
    groupBy: function(arr, key) {
        var groups = {};
        for (var i = 0; i < arr.length; i++) {
            var groupKey = arr[i][key];
            if (!groups[groupKey]) groups[groupKey] = [];
            groups[groupKey].push(arr[i]);
        }
        return groups;
    },

    /** Flatten one level of nested arrays. e.g. [[1,2],[3]] → [1,2,3] */
    flatten: function(arr) {
        var result = [];
        for (var i = 0; i < arr.length; i++) {
            if (arr[i] && typeof arr[i].length !== 'undefined') {
                for (var j = 0; j < arr[i].length; j++) result.push(arr[i][j]);
            } else {
                result.push(arr[i]);
            }
        }
        return result;
    },

    /** Return intersection of two arrays. */
    intersect: function(arr1, arr2) {
        var result = [];
        for (var i = 0; i < arr1.length; i++) {
            if (this.contains(arr2, arr1[i])) result.push(arr1[i]);
        }
        return result;
    },

    /** Return elements in arr1 that are NOT in arr2. */
    difference: function(arr1, arr2) {
        var result = [];
        for (var i = 0; i < arr1.length; i++) {
            if (!this.contains(arr2, arr1[i])) result.push(arr1[i]);
        }
        return result;
    },

    type: 'ArrayUtils'
};
