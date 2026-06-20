/**
 * Script Include: RESTMessageUtils
 * Category:       REST API > Outbound
 * Description:    Simplified wrapper for outbound REST calls using RESTMessageV2.
 * Client Callable: false
 * Scope:          Global
 *
 * Prerequisites:
 *   - Create an Outbound REST Message record under System Web Services > Outbound > REST Message
 *     OR use the direct URL method shown in callDirect().
 *
 * Usage:
 *   var rm = new RESTMessageUtils();
 *   // Named REST Message record:
 *   var resp = rm.call('My REST Message', 'Default GET', { param1: 'value1' });
 *   // Direct URL call:
 *   var resp = rm.callDirect('GET', 'https://api.example.com/data', {}, { Authorization: 'Bearer token' });
 */
var RESTMessageUtils = Class.create();
RESTMessageUtils.prototype = {
    initialize: function() {
        this.DEFAULT_TIMEOUT = 30000; // 30 seconds
    },

    /**
     * Execute a named REST Message (configured in the REST Message record).
     * @param {string} messageName    - Name of the REST Message record
     * @param {string} functionName   - Name of the HTTP Method/Function
     * @param {Object} [params]       - Key/value pairs for substitution variables
     * @returns {Object} { statusCode, body, headers, error }
     */
    call: function(messageName, functionName, params) {
        try {
            var rm = new sn_ws.RESTMessageV2(messageName, functionName);
            rm.setHttpTimeout(this.DEFAULT_TIMEOUT);
            if (params) {
                for (var key in params) {
                    if (params.hasOwnProperty(key)) {
                        rm.setStringParameterNoEscape(key, params[key]);
                    }
                }
            }
            var response = rm.execute();
            return this._parseResponse(response);
        } catch (ex) {
            gs.error('RESTMessageUtils.call error: ' + ex.message);
            return { statusCode: -1, body: null, headers: {}, error: ex.message };
        }
    },

    /**
     * Make a direct REST call without a named REST Message record.
     * @param {string} method        - HTTP method: GET, POST, PUT, PATCH, DELETE
     * @param {string} url           - Full endpoint URL
     * @param {Object} [body]        - Request body object (auto-serialized to JSON)
     * @param {Object} [headers]     - Additional HTTP headers
     * @returns {Object} { statusCode, body, headers, error }
     */
    callDirect: function(method, url, body, headers) {
        try {
            var rm = new sn_ws.RESTMessageV2();
            rm.setEndpoint(url);
            rm.setHttpMethod(method.toUpperCase());
            rm.setHttpTimeout(this.DEFAULT_TIMEOUT);
            rm.setRequestHeader('Content-Type', 'application/json');
            rm.setRequestHeader('Accept', 'application/json');
            if (headers) {
                for (var h in headers) {
                    if (headers.hasOwnProperty(h)) rm.setRequestHeader(h, headers[h]);
                }
            }
            if (body && (method.toUpperCase() === 'POST' || method.toUpperCase() === 'PUT' || method.toUpperCase() === 'PATCH')) {
                rm.setRequestBody(JSON.stringify(body));
            }
            var response = rm.execute();
            return this._parseResponse(response);
        } catch (ex) {
            gs.error('RESTMessageUtils.callDirect error: ' + ex.message);
            return { statusCode: -1, body: null, headers: {}, error: ex.message };
        }
    },

    /**
     * Execute a REST call asynchronously.
     * @param {string} messageName
     * @param {string} functionName
     * @param {Object} [params]
     * @returns {void}
     */
    callAsync: function(messageName, functionName, params) {
        try {
            var rm = new sn_ws.RESTMessageV2(messageName, functionName);
            if (params) {
                for (var key in params) {
                    if (params.hasOwnProperty(key)) {
                        rm.setStringParameterNoEscape(key, params[key]);
                    }
                }
            }
            rm.executeAsync();
        } catch (ex) {
            gs.error('RESTMessageUtils.callAsync error: ' + ex.message);
        }
    },

    /** Parse a RESTResponseV2 into a plain object. @private */
    _parseResponse: function(response) {
        var statusCode = response.getStatusCode();
        var rawBody    = response.getBody();
        var parsedBody = null;
        try {
            parsedBody = JSON.parse(rawBody);
        } catch (e) {
            parsedBody = rawBody; // return as raw string if not JSON
        }
        var headers = {};
        try {
            var headerStr = response.getHeaders();
            if (headerStr) headers = JSON.parse(headerStr);
        } catch (e) { /* ignore */ }
        return {
            statusCode: statusCode,
            body:       parsedBody,
            headers:    headers,
            error:      statusCode >= 400 ? ('HTTP ' + statusCode) : null
        };
    },

    /** Return true if the response indicates success (2xx status). */
    isSuccess: function(response) {
        return response && response.statusCode >= 200 && response.statusCode < 300;
    },

    type: 'RESTMessageUtils'
};
