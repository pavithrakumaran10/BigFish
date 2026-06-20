/**
 * Snippet Collection: Date & Time Operations
 * Context: Business Rules, Script Includes, Scheduled Jobs
 * Description: Common date/time patterns using GlideDateTime and GlideDuration.
 */

/* ---- Get current date/time ---- */
var now = new GlideDateTime();
gs.log('Now (internal): '  + now.getValue());           // yyyy-MM-dd HH:mm:ss
gs.log('Now (display): '   + now.getDisplayValue());    // user timezone format
gs.log('Date only: '       + now.getDate().getValue()); // yyyy-MM-dd
gs.log('Time only: '       + now.getTime().getValue()); // HH:mm:ss

/* ---- Set a specific date/time ---- */
var specific = new GlideDateTime();
specific.setDisplayValue('2025-12-31 23:59:00');
gs.log('Specific: ' + specific.getValue());

/* ---- Add/subtract time ---- */
var future = new GlideDateTime();
future.addDaysUTC(7);    // 7 days from now
future.addSeconds(3600); // +1 hour

var past = new GlideDateTime();
past.addDaysUTC(-30);    // 30 days ago

/* ---- Compare dates ---- */
var dt1 = new GlideDateTime();
dt1.setDisplayValue('2025-01-01 00:00:00');
var dt2 = new GlideDateTime();
dt2.setDisplayValue('2025-06-01 00:00:00');

gs.log('dt1 before dt2: ' + dt1.before(dt2));   // true
gs.log('dt1 after dt2:  ' + dt1.after(dt2));    // false
gs.log('equal:          ' + dt1.equals(dt2));   // false

/* ---- Calculate duration between two dates ---- */
var start = new GlideDateTime();
start.setDisplayValue('2025-01-01 08:00:00');
var end   = new GlideDateTime();
end.setDisplayValue('2025-01-03 10:30:00');

var diff = GlideDateTime.subtract(start, end);
gs.log('Days:    ' + diff.getDayPart());      // 2
gs.log('Hours:   ' + diff.getHourPart());    // 2
gs.log('Minutes: ' + diff.getMinutePart()); // 30

/* ---- GlideDuration examples ---- */
var dur = new GlideDuration();
dur.setValue('2 08:30:00'); // 2 days 8 hours 30 minutes
gs.log('Duration display: ' + dur.getDisplayValue());
gs.log('Total seconds: '   + dur.getDurationValue());

/* ---- Format date for display ---- */
var gdt = new GlideDateTime();
var gdf = new GlideDateTimeFormat(gdt);
gdf.setFormat('MMMM dd, yyyy');
gs.log('Formatted: ' + gdf.format()); // e.g. 'June 20, 2025'

/* ---- SLA breach detection ---- */
function isBreached(targetDateTime) {
    var target = new GlideDateTime(targetDateTime);
    return target.before(new GlideDateTime());
}

/* ---- Get start of current day and end of current day ---- */
var startOfDay = new GlideDateTime();
startOfDay.setHourUTC(0);
startOfDay.setMinuteUTC(0);
startOfDay.setSecondUTC(0);

var endOfDay = new GlideDateTime();
endOfDay.setHourUTC(23);
endOfDay.setMinuteUTC(59);
endOfDay.setSecondUTC(59);

/* ---- Working hours check ---- */
function isWithinBusinessHours(gdt, startHour, endHour) {
    /* CONFIGURE: startHour and endHour in UTC */
    var hour = gdt.getHourUTC();
    var dow  = gdt.getDayOfWeekLocalTime(); // 1=Sun, 7=Sat
    if (dow === 1 || dow === 7) return false; // weekend
    return (hour >= startHour && hour < endHour);
}
