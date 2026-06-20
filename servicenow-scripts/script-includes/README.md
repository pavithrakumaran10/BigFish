# Script Includes

Server-side JavaScript classes callable from Business Rules, Client Scripts (via GlideAjax), other Script Includes, REST APIs, and Flow Designer.

## Sub-categories

| Folder | Contents |
|---|---|
| `utils/` | String, Date, Array, Object helpers |
| `glide-record/` | CRUD wrappers for GlideRecord |
| `rest-api/` | Outbound REST message utilities |
| `user-group/` | User, role, and group helpers |
| `notification/` | Programmatic email/event helpers |
| `common/` | General-purpose catch-all utilities |

## How to Register a Script Include

1. Navigate to **System Definition > Script Includes**.
2. Click **New**.
3. Set **Name** to the class name (e.g., `StringUtils`).
4. Set **API Name** (auto-fills from Name).
5. Paste the script content.
6. Check **Client callable** if it will be called via GlideAjax.
7. Set **Application** and **Protection Policy** as needed.
8. Click **Submit**.

## Calling a Script Include

```javascript
// From a Business Rule or another Script Include
var util = new StringUtils();
var result = util.toCamelCase('hello world');

// From a Client Script via GlideAjax
var ga = new GlideAjax('StringUtils');
ga.addParam('sysparm_name', 'toCamelCase');
ga.addParam('input', 'hello world');
ga.getXMLAnswer(function(answer) { console.log(answer); });
```
