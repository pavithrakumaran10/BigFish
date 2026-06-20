/**
 * Client Script: FieldValidation
 * Type:          onSubmit
 * Table:         /* CONFIGURE: e.g. incident */
 * Description:   Client-side validation before form submission.
 *                Returns false to prevent saving when validation fails.
 *
 * Best practice: pair with a Before Business Rule for server-side enforcement.
 */

function onSubmit() {
    var valid = true;

    /* ---- Validate: due date must be in the future ---- */
    valid = valid && validateFutureDate('due_date', 'Due Date must be in the future.');

    /* ---- Validate: description must be at least 20 characters ---- */
    valid = valid && validateMinLength('description', 20, 'Description must be at least 20 characters.');

    /* ---- Validate: phone number format ---- */
    valid = valid && validatePattern(
        'phone',
        /^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$/,
        'Phone number is not in a valid format.'
    );

    /* ---- Validate: at least one attachment when state is Resolved ---- */
    if (g_form.getValue('state') === '6') {
        valid = valid && validateAttachment('At least one attachment is required when resolving an incident.');
    }

    return valid;
}

/** Check a date field is not in the past. */
function validateFutureDate(fieldName, message) {
    var val = g_form.getValue(fieldName);
    if (!val) return true; // empty is handled by mandatory check
    var fieldDate = new Date(val);
    var now = new Date();
    if (fieldDate <= now) {
        g_form.showFieldMsg(fieldName, message, 'error');
        return false;
    }
    g_form.hideFieldMsg(fieldName);
    return true;
}

/** Check a text field has a minimum character length. */
function validateMinLength(fieldName, minLen, message) {
    var val = (g_form.getValue(fieldName) || '').trim();
    if (val.length > 0 && val.length < minLen) {
        g_form.showFieldMsg(fieldName, message, 'error');
        return false;
    }
    g_form.hideFieldMsg(fieldName);
    return true;
}

/** Check a field value matches a regex pattern. */
function validatePattern(fieldName, regex, message) {
    var val = g_form.getValue(fieldName);
    if (!val) return true;
    if (!regex.test(val)) {
        g_form.showFieldMsg(fieldName, message, 'error');
        return false;
    }
    g_form.hideFieldMsg(fieldName);
    return true;
}

/** Check that the form has at least one attachment. */
function validateAttachment(message) {
    var attachCount = g_form.getAttachmentCount ? g_form.getAttachmentCount() : -1;
    if (attachCount === 0) {
        g_form.addErrorMessage(message);
        return false;
    }
    return true;
}
