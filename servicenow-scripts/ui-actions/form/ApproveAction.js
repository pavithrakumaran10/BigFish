/**
 * UI Action: Approve
 * Table:      /* CONFIGURE: e.g. u_approval_request */
 * Action name: approve_record
 * Form button: true
 * Condition:  current.state == 'pending' && gs.hasRole('approver_role') /* CONFIGURE */
 * Description: Move a record to 'Approved' state with an audit comment.
 *
 * How to create:
 *   1. Go to System Definition > UI Actions > New.
 *   2. Set Table, Name, Action name.
 *   3. Check 'Form button' and 'Client'.
 *   4. Paste this script.
 *   5. Set Condition to restrict visibility.
 */

/* ---- Client-side portion (runs in browser) ---- */
if (typeof window !== 'undefined') {
    function approve_record() { /* CONFIGURE: matches 'Action name' field */
        if (!confirm('Approve this request? This action cannot be undone.')) return false;
        g_form.save();
        gsftSubmit(null, g_form.getFormElement(), 'approve_record');
    }
}

/* ---- Server-side portion (runs on server) ---- */
if (typeof window === 'undefined') {
    (function() {
        /* Validate: record must still be pending */
        if (current.state.toString() !== 'pending') { /* CONFIGURE */
            gs.addErrorMessage('This record is no longer in Pending state and cannot be approved.');
            action.setRedirectURL(current);
            return;
        }

        /* Validate: approver must not be the requester */
        if (current.requested_by.toString() === gs.getUserID()) {
            gs.addErrorMessage('You cannot approve your own request.');
            action.setRedirectURL(current);
            return;
        }

        /* Apply approval */
        current.setValue('state',       'approved'); /* CONFIGURE */
        current.setValue('approved_by', gs.getUserID());
        current.setValue('approved_at', new GlideDateTime().getValue());
        current.setValue('work_notes',
            'Approved by ' + gs.getUserDisplayName() +
            ' on ' + new GlideDateTime().getDisplayValue()
        );
        current.update();

        /* Fire notification event */
        gs.eventQueue(
            'u_approval.approved', /* CONFIGURE: your event name */
            current,
            gs.getUserDisplayName(),
            current.getValue('number') || current.getUniqueValue()
        );

        gs.addInfoMessage('Record approved successfully.');
        action.setRedirectURL(current);
    })();
}
