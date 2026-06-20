/**
 * Snippet Collection: XML Handling
 * Context: Business Rules, Script Includes, Integration scripts
 * Description: Parse and build XML documents in ServiceNow using XMLDocument2.
 */

/* ---- Parse XML string ---- */
var xmlStr = '<?xml version="1.0" encoding="UTF-8"?>' +
    '<response>' +
    '  <status>success</status>' +
    '  <incident number="INC001" priority="1">' +
    '    <description>Network down</description>' +
    '    <assignee>Alice Smith</assignee>' +
    '  </incident>' +
    '</response>';

var doc = new XMLDocument2();
doc.parseXML(xmlStr);

// Get a single element's text content
var status = doc.getNode('//response/status');
gs.log('Status: ' + (status ? status.getTextContent() : 'n/a'));

// Get attribute value
var incNode = doc.getNode('//incident');
if (incNode) {
    gs.log('Number:   ' + incNode.getAttribute('number'));   // INC001
    gs.log('Priority: ' + incNode.getAttribute('priority')); // 1
}

// Iterate over multiple nodes
var nodeIter = doc.getNodes('//incident');
while (nodeIter.hasNext()) {
    var node = nodeIter.next();
    gs.log('Incident: ' + node.getAttribute('number'));
    var descNode = doc.getNode('description', node);
    if (descNode) gs.log('Desc: ' + descNode.getTextContent());
}

/* ---- Build an XML string ---- */
var builder = new XMLDocument2();
var root    = builder.createElement('request');
root.setAttribute('version', '1.0');

var incEl  = builder.createElement('incident', root);
var numEl  = builder.createElement('number',   incEl);
numEl.setTextContent('INC002');
var prioEl = builder.createElement('priority', incEl);
prioEl.setTextContent('2');

gs.log(builder.toString());

/* ---- Validate that XML is well-formed ---- */
function isValidXML(xmlString) {
    try {
        var testDoc = new XMLDocument2();
        testDoc.parseXML(xmlString);
        return testDoc.isValid();
    } catch (e) {
        return false;
    }
}

/* ---- Extract all text values by tag name ---- */
function extractByTag(xmlStr, tagName) {
    var values = [];
    var doc2   = new XMLDocument2();
    doc2.parseXML(xmlStr);
    var nodes = doc2.getNodes('//' + tagName);
    while (nodes.hasNext()) {
        values.push(nodes.next().getTextContent());
    }
    return values;
}
