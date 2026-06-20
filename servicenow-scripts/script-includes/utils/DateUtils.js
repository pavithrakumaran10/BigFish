/**
 * Script Include: DateUtils
 * Category:       Utilities > Date/Time
 * Description:    Date and time helpers using GlideDateTime for ServiceNow.
 * Client Callable: false
 * Scope:          Global
 *
 * Usage:
 *   var du = new DateUtils();
 *   var now   = du.now();
 *   var diff  = du.daysBetween('2024-01-01', '2024-03-15');
 *   var biz   = du.addBusinessDays(new GlideDateTime(), 5);
 */
var DateUtils = Class.create();
DateUtils.prototype = {
    initialize: function() {},

    /** Return current GlideDateTime. */
    now: function() {
        return new GlideDateTime();
    },

    /** Return today's date as a string in yyyy-MM-dd format. */
    todayString: function() {
        var gdt = new GlideDateTime();
        return gdt.getDate().getValue();
    },

    /**
     * Return the number of calendar days between two dates.
     * @param {string} startDate - yyyy-MM-dd or GlideDateTime
     * @param {string} endDate   - yyyy-MM-dd or GlideDateTime
     * @returns {number}
     */
    daysBetween: function(startDate, endDate) {
        var start = new GlideDateTime();
        var end   = new GlideDateTime();
        start.setDisplayValue(String(startDate));
        end.setDisplayValue(String(endDate));
        var diff = new GlideDuration(Math.abs(end.getNumericValue() - start.getNumericValue()));
        return diff.getDayPart();
    },

    /**
     * Add a number of calendar days to a GlideDateTime and return the result.
     * @param {GlideDateTime} gdt
     * @param {number} days - can be negative to subtract
     * @returns {GlideDateTime}
     */
    addDays: function(gdt, days) {
        var result = new GlideDateTime(gdt);
        result.addDaysUTC(days);
        return result;
    },

    /**
     * Add business days (Mon–Fri) to a GlideDateTime.
     * Does NOT account for holidays — extend to add holiday logic.
     * @param {GlideDateTime} gdt
     * @param {number} days
     * @returns {GlideDateTime}
     */
    addBusinessDays: function(gdt, days) {
        var result = new GlideDateTime(gdt);
        var added  = 0;
        var direction = days >= 0 ? 1 : -1;
        var remaining = Math.abs(days);
        while (remaining > 0) {
            result.addDaysUTC(direction);
            var dow = result.getDayOfWeekLocalTime(); // 1=Sun,2=Mon,...,7=Sat
            if (dow !== 1 && dow !== 7) {             // skip Sat & Sun
                remaining--;
            }
        }
        return result;
    },

    /**
     * Check whether a GlideDateTime falls on a weekend.
     * @param {GlideDateTime} gdt
     * @returns {boolean}
     */
    isWeekend: function(gdt) {
        var dow = gdt.getDayOfWeekLocalTime();
        return dow === 1 || dow === 7; // Sunday=1, Saturday=7
    },

    /**
     * Format a GlideDateTime to a human-readable string.
     * @param {GlideDateTime} gdt
     * @param {string} format - e.g. 'MM/dd/yyyy HH:mm:ss'
     * @returns {string}
     */
    format: function(gdt, format) {
        if (!gdt) return '';
        var gdf = new GlideDateTimeFormat(gdt);
        gdf.setFormat(format || 'MM/dd/yyyy');
        return gdf.format();
    },

    /**
     * Return true if the given date is in the past.
     * @param {GlideDateTime} gdt
     * @returns {boolean}
     */
    isPast: function(gdt) {
        return gdt.before(new GlideDateTime());
    },

    /**
     * Return true if the given date is in the future.
     * @param {GlideDateTime} gdt
     * @returns {boolean}
     */
    isFuture: function(gdt) {
        return gdt.after(new GlideDateTime());
    },

    /**
     * Return start of day (midnight) for a GlideDateTime.
     * @param {GlideDateTime} gdt
     * @returns {GlideDateTime}
     */
    startOfDay: function(gdt) {
        var result = new GlideDateTime(gdt);
        result.setHourUTC(0);
        result.setMinuteUTC(0);
        result.setSecondUTC(0);
        return result;
    },

    /**
     * Return end of day (23:59:59) for a GlideDateTime.
     * @param {GlideDateTime} gdt
     * @returns {GlideDateTime}
     */
    endOfDay: function(gdt) {
        var result = new GlideDateTime(gdt);
        result.setHourUTC(23);
        result.setMinuteUTC(59);
        result.setSecondUTC(59);
        return result;
    },

    type: 'DateUtils'
};
