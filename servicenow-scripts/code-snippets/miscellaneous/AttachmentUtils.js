/**
 * Snippet Collection: Attachment Utilities
 * Context: Business Rules, Script Includes, Scheduled Jobs
 * Description: Read, create, copy, and delete attachments programmatically.
 */

/* ---- Get all attachments for a record ---- */
function getAttachments(tableName, recordSysId) {
    var attachments = [];
    var gr = new GlideRecord('sys_attachment');
    gr.addQuery('table_name', tableName);
    gr.addQuery('table_sys_id', recordSysId);
    gr.query();
    while (gr.next()) {
        attachments.push({
            sys_id:        gr.getUniqueValue(),
            file_name:     gr.getValue('file_name'),
            content_type:  gr.getValue('content_type'),
            size_bytes:    gr.getValue('size_bytes'),
            created_on:    gr.getValue('sys_created_on')
        });
    }
    return attachments;
}

/* ---- Read attachment content as a string ---- */
function readAttachmentContent(attachmentSysId) {
    var sa = new GlideSysAttachment();
    var gr = new GlideRecord('sys_attachment');
    if (!gr.get(attachmentSysId)) return null;
    return sa.get(gr); // returns the file content as a string
}

/* ---- Create a text attachment on a record ---- */
function createTextAttachment(tableName, recordSysId, fileName, content) {
    var sa = new GlideSysAttachment();
    return sa.write(
        tableName,
        recordSysId,
        fileName,
        'text/plain',
        content
    );
}

/* ---- Create a JSON attachment ---- */
function createJSONAttachment(tableName, recordSysId, fileName, obj) {
    var content = JSON.stringify(obj, null, 2);
    var sa = new GlideSysAttachment();
    return sa.write(tableName, recordSysId, fileName, 'application/json', content);
}

/* ---- Copy attachment from one record to another ---- */
function copyAttachment(attachmentSysId, targetTable, targetSysId) {
    var source = new GlideRecord('sys_attachment');
    if (!source.get(attachmentSysId)) return null;

    var sa      = new GlideSysAttachment();
    var content = sa.get(source);

    return sa.write(
        targetTable,
        targetSysId,
        source.getValue('file_name'),
        source.getValue('content_type'),
        content
    );
}

/* ---- Delete an attachment by sys_id ---- */
function deleteAttachment(attachmentSysId) {
    var gr = new GlideRecord('sys_attachment');
    if (gr.get(attachmentSysId)) {
        gr.deleteRecord();
        return true;
    }
    return false;
}

/* ---- Delete all attachments from a record ---- */
function deleteAllAttachments(tableName, recordSysId) {
    var gr = new GlideRecord('sys_attachment');
    gr.addQuery('table_name',   tableName);
    gr.addQuery('table_sys_id', recordSysId);
    gr.deleteMultiple();
}

/* ---- Get attachment count for a record ---- */
function getAttachmentCount(tableName, recordSysId) {
    var ga = new GlideAggregate('sys_attachment');
    ga.addQuery('table_name',   tableName);
    ga.addQuery('table_sys_id', recordSysId);
    ga.addAggregate('COUNT');
    ga.query();
    if (ga.next()) return parseInt(ga.getAggregate('COUNT'), 10);
    return 0;
}
