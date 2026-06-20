/**
 * Snippet Collection: ACL & Security Utilities
 * Context: ACL scripts (System Security > Access Control), Business Rules, Script Includes
 * Description: Common access control and security patterns in ServiceNow.
 */

/* ============================================================
   ACL CONDITION SCRIPTS
   These go in the 'Condition' or 'Script' field of an ACL record.
   Navigate to: System Security > Access Control (ACL) > New
   ============================================================ */

/* ---- Allow only the record's creator or an admin ---- */
// ACL Script field:
(function() {
    return (current.sys_created_by.toString() === gs.getUserName()) ||
           gs.hasRole('admin');
})();

/* ---- Allow only members of the assignment group ---- */
(function() {
    if (gs.hasRole('admin')) return true;
    var groupSysId = current.assignment_group.toString();
    if (!groupSysId) return false;
    var gr = new GlideRecord('sys_user_grmember');
    gr.addQuery('user',  gs.getUserID());
    gr.addQuery('group', groupSysId);
    gr.setLimit(1);
    gr.query();
    return gr.next();
})();

/* ---- Allow write only in specific states ---- */
(function() {
    var editableStates = ['1', '2', '3']; // New, In Progress, On Hold
    return editableStates.indexOf(current.state.toString()) !== -1 ||
           gs.hasRole('admin');
})();

/* ---- Time-based access: only during business hours ---- */
(function() {
    if (gs.hasRole('admin')) return true;
    var now = new GlideDateTime();
    var hour = now.getHourOfDayLocalTime();
    var dow  = now.getDayOfWeekLocalTime(); // 1=Sun, 7=Sat
    return (dow >= 2 && dow <= 6) && (hour >= 8 && hour < 18); /* CONFIGURE */
})();


/* ============================================================
   ROLE & PERMISSION HELPERS
   ============================================================ */

/* ---- Check current user has ANY of the listed roles ---- */
function hasAnyRole(roles) {
    for (var i = 0; i < roles.length; i++) {
        if (gs.hasRole(roles[i])) return true;
    }
    return false;
}
// Usage:
// if (hasAnyRole(['itil', 'itil_admin', 'admin'])) { ... }

/* ---- Check current user has ALL of the listed roles ---- */
function hasAllRoles(roles) {
    for (var i = 0; i < roles.length; i++) {
        if (!gs.hasRole(roles[i])) return false;
    }
    return true;
}

/* ---- Elevate to admin for a block (use sparingly) ---- */
function runAsAdmin(fn) {
    var wasElevated = gs.hasRole('admin');
    if (!wasElevated) gs.setProperty('glide.su_elevate', 'true');
    try {
        return fn();
    } finally {
        if (!wasElevated) gs.setProperty('glide.su_elevate', 'false');
    }
}


/* ============================================================
   DATA SANITIZATION
   ============================================================ */

/* ---- Prevent script injection: escape HTML before storing user input ---- */
function sanitizeInput(input) {
    if (!input) return '';
    return String(input)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/* ---- Validate sys_id format (32-char hex) ---- */
function isValidSysId(sysId) {
    if (!sysId) return false;
    return /^[0-9a-f]{32}$/i.test(String(sysId));
}

/* ---- Validate email format ---- */
function isValidEmail(email) {
    if (!email) return false;
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(email));
}

/* ---- Strip HTML tags from a string ---- */
function stripHTML(html) {
    if (!html) return '';
    return String(html).replace(/<[^>]*>/g, '');
}
