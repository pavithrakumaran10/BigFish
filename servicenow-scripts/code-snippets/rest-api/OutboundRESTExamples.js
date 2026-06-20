/**
 * Snippet Collection: Outbound REST API Examples
 * Context: Business Rules (async), Script Includes, Scheduled Jobs
 * Description: Patterns for calling external REST APIs from ServiceNow.
 */

/* ---- Pattern 1: Named REST Message (configured in UI) ---- */
var rm1 = new sn_ws.RESTMessageV2('My External API', 'GET User'); /* CONFIGURE */
rm1.setStringParameterNoEscape('user_id', 'ext-12345');
rm1.setHttpTimeout(10000);
var resp1 = rm1.execute();
if (resp1.getStatusCode() === 200) {
    var user = JSON.parse(resp1.getBody());
    gs.log('User name: ' + user.name);
}

/* ---- Pattern 2: Direct HTTP call (no pre-configured record) ---- */
var rm2 = new sn_ws.RESTMessageV2();
rm2.setEndpoint('https://jsonplaceholder.typicode.com/posts/1'); /* CONFIGURE */
rm2.setHttpMethod('GET');
rm2.setRequestHeader('Accept', 'application/json');
rm2.setHttpTimeout(15000);
var resp2 = rm2.execute();
gs.log('Status: ' + resp2.getStatusCode());
gs.log('Body: '   + resp2.getBody());

/* ---- Pattern 3: POST with JSON body ---- */
var payload = {
    title:  'ServiceNow Integration',
    body:   'Auto-created from ServiceNow',
    userId: 1
};
var rm3 = new sn_ws.RESTMessageV2();
rm3.setEndpoint('https://jsonplaceholder.typicode.com/posts'); /* CONFIGURE */
rm3.setHttpMethod('POST');
rm3.setRequestHeader('Content-Type', 'application/json');
rm3.setRequestHeader('Authorization', 'Bearer ' + gs.getProperty('u_api_token')); /* CONFIGURE */
rm3.setRequestBody(JSON.stringify(payload));
var resp3 = rm3.execute();
var created = JSON.parse(resp3.getBody());
gs.log('Created ID: ' + created.id);

/* ---- Pattern 4: Async REST call (non-blocking) ---- */
var rm4 = new sn_ws.RESTMessageV2('Webhook Notification', 'POST'); /* CONFIGURE */
rm4.setStringParameterNoEscape('incident_number', current.getValue('number'));
rm4.executeAsync(); // fire and forget

/* ---- Pattern 5: Handle errors and retry ---- */
function callWithRetry(url, maxRetries) {
    var retries  = 0;
    var response = null;
    while (retries < maxRetries) {
        try {
            var rm = new sn_ws.RESTMessageV2();
            rm.setEndpoint(url);
            rm.setHttpMethod('GET');
            rm.setHttpTimeout(10000);
            response = rm.execute();
            if (response.getStatusCode() < 500) break; // success or client error
        } catch (ex) {
            gs.warn('REST retry ' + retries + ': ' + ex.message);
        }
        retries++;
    }
    return response;
}

/* ---- Pattern 6: Parse paginated API response ---- */
function fetchAllPages(baseUrl, token) {
    var allItems = [];
    var page     = 1;
    var hasMore  = true;
    while (hasMore) {
        var rm = new sn_ws.RESTMessageV2();
        rm.setEndpoint(baseUrl + '?page=' + page + '&per_page=100');
        rm.setHttpMethod('GET');
        rm.setRequestHeader('Authorization', 'Bearer ' + token);
        var resp = rm.execute();
        var data = JSON.parse(resp.getBody());
        if (data.items && data.items.length > 0) {
            for (var i = 0; i < data.items.length; i++) allItems.push(data.items[i]);
            page++;
        } else {
            hasMore = false;
        }
    }
    return allItems;
}
