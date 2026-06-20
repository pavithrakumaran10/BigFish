/**
 * UI Action: Bulk Close Resolved Incidents
 * Table:      /* CONFIGURE: e.g. incident */
 * Action name: bulk_close_incidents
 * List button: true
 * Condition:  gs.hasRole('itil') /* CONFIGURE */
 * Description: Close all selected resolved incidents from a list view.
 *              Iterates over the checked rows and updates each record.
 *
 * Note: List actions receive 'sysparm_checked_items' with comma-separated sys_ids.
 */

/* ---- Client-side: confirm before bulk action ---- */
if (typeof window !== 'undefined') {
    function bulk_close_incidents() {
        var checkedItems = g_list.getChecked();
        if (!checkedItems) {
            alert('Please select at least one record.');
            return false;
        }
        var count = checkedItems.split(',').length;
        if (!confirm('Close ' + count + ' selected incident(s)?')) return false;
        return true;
    }
}

/* ---- Server-side: process the selected records ---- */
if (typeof window === 'undefined') {
    (function() {
        var checkedItems = g_list ? g_list.getChecked() : '';
        /* Retrieve selected sys_ids from the form parameter */
        var items = String(current.getUniqueValue()); /* fallback for single record */
        try {
            var param = g_form ? g_form.getParameter('sysparm_checked_items') : '';
            if (param) items = param;
        } catch (e) { /* ignore */ }

        if (!items) {
            gs.addErrorMessage('No records selected.');
            return;
        }

        var sysIds    = items.split(',');
        var closed    = 0;
        var skipped   = 0;
        var closeNote = 'Bulk closed by ' + gs.getUserDisplayName() + ' on ' + new GlideDateTime().getDisplayValue();

        for (var i = 0; i < sysIds.length; i++) {
            var gr = new GlideRecord('incident'); /* CONFIGURE */
            if (!gr.get(sysIds[i].trim())) { skipped++; continue; }

            /* Only close records that are in Resolved state */
            if (gr.getValue('state') !== '6') { /* CONFIGURE: '6' = Resolved */
                skipped++;
                continue;
            }

            gr.setValue('state',       '7');      /* CONFIGURE: '7' = Closed */
            gr.setValue('close_notes', closeNote);
            gr.setValue('closed_at',   new GlideDateTime().getValue());
            gr.setValue('closed_by',   gs.getUserID());
            gr.update();
            closed++;
        }

        gs.addInfoMessage(
            'Bulk close complete: ' + closed + ' closed, ' + skipped + ' skipped (not in Resolved state).'
        );
    })();
}
