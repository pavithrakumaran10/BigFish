/**
 * Client Script: ShowHideFields
 * Type:          onLoad / onChange
 * Table:         /* CONFIGURE: e.g. incident */
 * Description:   Show or hide form fields based on field values or user roles.
 *
 * How to apply:
 *   1. Create a Client Script on your target table.
 *   2. Set Type to 'onLoad' for initial state, or 'onChange' + Field Name for reactive behavior.
 *   3. Paste this script and customize the CONFIGURE sections.
 */

/* ---- onLoad example: hide 'resolution_notes' until state = Resolved ---- */
function onLoad() {
    setResolutionFieldVisibility();
}

/* ---- onChange example: watch the 'state' field ---- */
function onChange(control, oldValue, newValue, isLoading) {
    if (isLoading) return; // skip during initial form load
    setResolutionFieldVisibility();
}

function setResolutionFieldVisibility() {
    /* CONFIGURE: adjust field name and value that triggers visibility */
    var state = g_form.getValue('state');
    var isResolved = (state === '6'); // 6 = Resolved in ITSM

    g_form.setVisible('close_code',      isResolved);
    g_form.setVisible('close_notes',     isResolved);
    g_form.setVisible('resolved_at',     isResolved);
    g_form.setVisible('resolved_by',     isResolved);

    /* Make resolution fields mandatory when resolved */
    g_form.setMandatory('close_code',  isResolved);
    g_form.setMandatory('close_notes', isResolved);
}

/* ---- Utility: show/hide multiple fields at once ---- */
function setFieldsVisible(fieldNames, visible) {
    for (var i = 0; i < fieldNames.length; i++) {
        g_form.setVisible(fieldNames[i], visible);
    }
}

/* ---- Utility: hide a field section/tab ---- */
function hideSectionByCaption(caption) {
    /* CONFIGURE: match the section header label exactly */
    var sections = document.querySelectorAll('.section-head');
    for (var i = 0; i < sections.length; i++) {
        if (sections[i].textContent.trim() === caption) {
            sections[i].closest('tbody').style.display = 'none';
        }
    }
}
