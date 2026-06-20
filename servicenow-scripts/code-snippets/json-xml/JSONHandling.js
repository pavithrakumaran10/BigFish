/**
 * Snippet Collection: JSON Handling
 * Context: Business Rules, Script Includes, REST API scripts
 * Description: Patterns for parsing, building, and manipulating JSON in ServiceNow.
 */

/* ---- Parse a JSON string ---- */
var raw = '{"name":"Alice","age":30,"roles":["admin","user"]}';
try {
    var data = JSON.parse(raw);
    gs.log('Name: ' + data.name);          // Alice
    gs.log('First role: ' + data.roles[0]); // admin
} catch (e) {
    gs.error('JSON parse error: ' + e.message);
}

/* ---- Serialize an object to JSON ---- */
var incident = {
    number:      'INC0012345',
    state:       'In Progress',
    priority:    1,
    tags:        ['network', 'critical'],
    resolved:    false,
    resolved_at: null
};
var jsonStr = JSON.stringify(incident);
gs.log(jsonStr);

/* ---- Pretty-print JSON (for logs or work notes) ---- */
gs.log(JSON.stringify(incident, null, 2));

/* ---- Safely get a nested property ---- */
function safeGet(obj, path, defaultVal) {
    var parts = path.split('.');
    var curr  = obj;
    for (var i = 0; i < parts.length; i++) {
        if (curr === null || curr === undefined) return defaultVal;
        curr = curr[parts[i]];
    }
    return (curr !== undefined && curr !== null) ? curr : defaultVal;
}
var resp = { data: { user: { email: 'alice@example.com' } } };
gs.log(safeGet(resp, 'data.user.email', 'no-email'));   // alice@example.com
gs.log(safeGet(resp, 'data.user.phone', 'no-phone'));   // no-phone

/* ---- Merge two objects (shallow) ---- */
function shallowMerge(target, source) {
    var result = {};
    for (var k in target) { if (target.hasOwnProperty(k)) result[k] = target[k]; }
    for (var k2 in source)  { if (source.hasOwnProperty(k2)) result[k2] = source[k2]; }
    return result;
}
var defaults = { urgency: 3, impact: 3, category: 'inquiry' };
var incoming = { urgency: 1, category: 'hardware' };
var merged   = shallowMerge(defaults, incoming);
// { urgency: 1, impact: 3, category: 'hardware' }

/* ---- Filter keys from an object ---- */
function pickKeys(obj, keys) {
    var result = {};
    for (var i = 0; i < keys.length; i++) {
        if (obj.hasOwnProperty(keys[i])) result[keys[i]] = obj[keys[i]];
    }
    return result;
}
var full    = { number: 'INC001', state: '2', internal_id: 'x99', priority: '1' };
var trimmed = pickKeys(full, ['number', 'state', 'priority']);
// { number: 'INC001', state: '2', priority: '1' }

/* ---- Convert GlideRecord to plain JSON object ---- */
function grToJSON(gr, fields) {
    var obj = { sys_id: gr.getUniqueValue() };
    for (var i = 0; i < fields.length; i++) {
        obj[fields[i]] = gr.getValue(fields[i]);
    }
    return obj;
}
var inc = new GlideRecord('incident');
inc.setLimit(1);
inc.query();
if (inc.next()) {
    var incJSON = grToJSON(inc, ['number', 'short_description', 'state', 'priority']);
    gs.log(JSON.stringify(incJSON));
}
