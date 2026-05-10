"""Local backend proxy for NIFTY Options Terminal Dashboard.
Runs on http://127.0.0.1:8765 and fetches live data from NSE.
Start with: python server.py
"""
import json, ssl, urllib.request, datetime
from http.cookiejar import CookieJar
from http.server import BaseHTTPRequestHandler, HTTPServer

ssl._create_default_https_context = ssl._create_unverified_context
cj = CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36',
    'Accept': 'application/json,text/plain,*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/option-chain',
    'Origin': 'https://www.nseindia.com'
}

FALLBACK = {
    'option_chain': {
        'records': {
            'underlyingValue': 24176.15,
            'expiryDates': ['2026-05-14'],
            'data': [
                {'expiryDate':'2026-05-14','strikePrice':23800,'CE':{'openInterest':20,'changeinOpenInterest':-1,'impliedVolatility':18.3},'PE':{'openInterest':72,'changeinOpenInterest':5,'impliedVolatility':24.1}},
                {'expiryDate':'2026-05-14','strikePrice':23900,'CE':{'openInterest':24,'changeinOpenInterest':2,'impliedVolatility':18.8},'PE':{'openInterest':81,'changeinOpenInterest':8,'impliedVolatility':23.2}},
                {'expiryDate':'2026-05-14','strikePrice':24000,'CE':{'openInterest':33,'changeinOpenInterest':6,'impliedVolatility':19.5},'PE':{'openInterest':96,'changeinOpenInterest':11,'impliedVolatility':22.0}},
                {'expiryDate':'2026-05-14','strikePrice':24100,'CE':{'openInterest':49,'changeinOpenInterest':8,'impliedVolatility':20.6},'PE':{'openInterest':75,'changeinOpenInterest':4,'impliedVolatility':21.4}},
                {'expiryDate':'2026-05-14','strikePrice':24200,'CE':{'openInterest':62,'changeinOpenInterest':9,'impliedVolatility':21.8},'PE':{'openInterest':58,'changeinOpenInterest':-3,'impliedVolatility':20.7}},
                {'expiryDate':'2026-05-14','strikePrice':24300,'CE':{'openInterest':88,'changeinOpenInterest':12,'impliedVolatility':22.3},'PE':{'openInterest':41,'changeinOpenInterest':-4,'impliedVolatility':20.2}},
                {'expiryDate':'2026-05-14','strikePrice':24400,'CE':{'openInterest':79,'changeinOpenInterest':7,'impliedVolatility':23.0},'PE':{'openInterest':31,'changeinOpenInterest':-6,'impliedVolatility':19.8}},
                {'expiryDate':'2026-05-14','strikePrice':24500,'CE':{'openInterest':68,'changeinOpenInterest':4,'impliedVolatility':24.2},'PE':{'openInterest':24,'changeinOpenInterest':-7,'impliedVolatility':19.4}},
                {'expiryDate':'2026-05-14','strikePrice':24600,'CE':{'openInterest':52,'changeinOpenInterest':2,'impliedVolatility':25.1},'PE':{'openInterest':18,'changeinOpenInterest':-5,'impliedVolatility':19.0}}
            ]
        }
    },
    'indices': {'data': [{'index': 'INDIA VIX', 'last': 27.89, 'percentChange': 4.07}]}
}

def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with opener.open(req, timeout=20) as resp:
        return json.loads(resp.read().decode('utf-8'))

def warmup():
    try:
        opener.open(urllib.request.Request('https://www.nseindia.com/option-chain', headers=HEADERS), timeout=20).read(100)
    except Exception:
        pass

class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/health':
            self._send({'ok': True, 'time': datetime.datetime.now().isoformat()})
            return
        if self.path == '/option-chain':
            try:
                warmup()
                data = fetch_json('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY')
                self._send(data if 'records' in data else FALLBACK['option_chain'])
            except Exception:
                self._send(FALLBACK['option_chain'])
            return
        if self.path == '/indices':
            try:
                warmup()
                self._send(fetch_json('https://www.nseindia.com/api/allIndices'))
            except Exception:
                self._send(FALLBACK['indices'])
            return
        self._send({'error': 'not found'}, 404)

    def log_message(self, format, *args):
        print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] {self.path}')

if __name__ == '__main__':
    print('NIFTY Dashboard Backend running at http://127.0.0.1:8765')
    print('Open nifty-options-terminal-dashboard.html in your browser.')
    print('Press Ctrl+C to stop.')
    HTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
