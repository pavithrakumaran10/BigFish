/**
 * UI Action: Reject
 * Table:      /* CONFIGURE: e.g. u_approval_request */
 * Action name: reject_record
 * Form button: true
 * Condition:  current.state == 'pending' && gs.hasRole('approver_role') /* CONFIGURE */
 * Description: Move a record to 'Rejected' state and capture a rejection reason.
 */

/* ---- Client-side portion ---- */
if (typeof window !== 'undefined') {
    function reject_record() {
        var reason = prompt('Please enter a rejection reason (required):');
        if (!reason || reason.trim() === '') {
            alert('A rejection reason is required.');
            return false;
        }
        g_form.setValue('u_rejection_reason', reason.trim()); /* CONFIGURE: ensure field exists */
        gsftSubmit(null, g_form.getFormElement(), 'reject_record');
    }
}

/* ---- Server-side portion ---- */
if (typeof window === 'undefined') {
    (function() {
        /* Validate state */
        if (current.state.toString() !== 'pending') { /* CONFIGURE */
            gs.addErrorMessage('This record is no longer in Pending state.');
            action.setRedirectURL(current);
            return;
        }

        /* Validate rejection reason was provided */
        if (current.u_rejection_reason.nil()) {
            gs.addErrorMessage('A rejection reason is required.');
            action.setRedirectURL(current);
            return;
        }

        /* Apply rejection */
        current.setValue('state',       'rejected'); /* CONFIGURE */
        current.setValue('rejected_by', gs.getUserID());
        current.setValue('rejected_at', new GlideDateTime().getValue());
        current.setValue('work_notes',
            'Rejected by ' + gs.getUserDisplayName() + '.\n' +
            'Reason: ' + current.getValue('u_rejection_reason')
        );
        current.update();

        /* Fire notification event */
        gs.eventQueue(
            'u_approval.rejected', /* CONFIGURE */
            current,
            gs.getUserDisplayName(),
            current.getValue('u_rejection_reason')
        );

        gs.addInfoMessage('Record has been rejected.');
        action.setRedirectURL(current);
    })();
}
