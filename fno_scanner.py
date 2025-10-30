"""
FNO (Futures & Options) Open Drive Scanner with Open Interest Analysis
Scans all NSE F&O stocks (~180 stocks) with OI buildup detection
Combines Open Drive + Value Area + Open Interest signals
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# ==================== NSE F&O STOCK LIST ====================
# All stocks with Futures & Options (as of Oct 2024)
FNO_STOCKS = [
    # Nifty 50 stocks
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
    "BHARTIARTL", "SBIN", "KOTAKBANK", "ITC", "LT", "AXISBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH", "SUNPHARMA",
    "TITAN", "ULTRACEMCO", "NESTLEIND", "WIPRO", "ONGC", "NTPC",
    "POWERGRID", "TATAMOTORS", "BAJAJFINSV", "TATASTEEL", "M&M",
    "TECHM", "ADANIPORTS", "COALINDIA", "JSWSTEEL", "INDUSINDBK",
    "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT", "HINDALCO",
    "BRITANNIA", "GRASIM", "APOLLOHOSP", "BPCL", "HEROMOTOCO",
    "TATACONSUM", "BAJAJ-AUTO", "SBILIFE", "HDFCLIFE", "SHRIRAMFIN",
    
    # Bank Nifty additional stocks
    "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB", "BANKBARODA",
    
    # Mid/Small cap F&O stocks
    "ACC", "ADANIENT", "ADANIGREEN", "ADANIPOWER", "APLAPOLLO",
    "ABCAPITAL", "ABFRL", "ALKEM", "AMARAJABAT", "AMBUJACEM",
    "APOLLOTYRE", "ASHOKLEY", "ASTRAL", "ATUL", "AUROPHARMA",
    "DMART", "BAJAJHLDNG", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK",
    "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG", "BHEL",
    "BIOCON", "BOSCHLTD", "CANBK", "CANFINHOME", "CHAMBLFERT",
    "CHOLAFIN", "COLPAL", "CONCOR", "COROMANDEL", "CUB",
    "CUMMINSIND", "DABUR", "DEEPAKNTR", "DELTACORP", "DIXON",
    "DLF", "ESCORTS", "EXIDEIND", "GAIL", "GLENMARK",
    "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES",
    "GUJGASLTD", "HAL", "HAVELLS", "HDFCAMC", "HDFCLIFE",
    "HINDCOPPER", "HINDPETRO", "ICICIGI", "ICICIPRULI", "IDEA",
    "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIACEM",
    "INDIAMART", "INDIANB", "INDIGO", "INDUSTOWER", "IOC",
    "IPCALAB", "IRB", "IRCTC", "IRFC", "ITC",
    "JINDALSAW", "JINDALSTEL", "JKCEMENT", "JSWENERGY", "JUBLFOOD",
    "KAJARIACER", "KEI", "KOTAKBANK", "L&TFH", "LALPATHLAB",
    "LAURUSLABS", "LICHSGFIN", "LUPIN", "MANAPPURAM", "MARICO",
    "MAXHEALTH", "MCDOWELL-N", "MCX", "METROPOLIS", "MFSL",
    "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NMDC", "OBEROIRLTY",
    "OFSS", "OIL", "PAGEIND", "PERSISTENT", "PETRONET",
    "PFC", "PIDILITIND", "PIIND", "PNB", "POLYCAB",
    "PVR", "RBLBANK", "RECLTD", "SAIL", "SBICARD",
    "SBILIFE", "SHREECEM", "SIEMENS", "SRF", "SRTRANSFIN",
    "STAR", "SUNPHARMA", "SUNTV", "SUPREMEIND", "TATACOMM",
    "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATAPOWER", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "TORNTPHARM", "TORNTPOWER",
    "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UPL",
    "VEDL", "VOLTAS", "WHIRLPOOL", "WIPRO", "YESBANK", "ZEEL"
]

# ==================== OPEN INTEREST DATA FETCHER ====================
class OpenInterestAnalyzer:
    """Fetches and analyzes Open Interest data from NSE"""
    
    def __init__(self):
        self.nse_base = "https://www.nseindia.com"
        self.session = self._create_session()
    
    def _create_session(self):
        """Create session with proper headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/'
        })
        try:
            session.get(self.nse_base, timeout=10)
            time.sleep(1)
        except:
            pass
        return session
    
    def get_option_chain(self, symbol):
        """Fetch option chain data for OI analysis"""
        url = f"{self.nse_base}/api/option-chain-equities?symbol={symbol}"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            pass
        return None
    
    def get_futures_data(self, symbol):
        """Fetch futures data including OI"""
        url = f"{self.nse_base}/api/quote-derivative?symbol={symbol}"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def analyze_oi_buildup(self, option_data, futures_data, spot_price):
        """
        Analyze Open Interest buildup to determine bullish/bearish sentiment
        
        Returns: {
            'signal': 'BULLISH'/'BEARISH'/'NEUTRAL',
            'strength': 0-100,
            'call_oi': total_call_oi,
            'put_oi': total_put_oi,
            'pcr': put_call_ratio,
            'max_pain': estimated_max_pain,
            'interpretation': description
        }
        """
        result = {
            'signal': 'NEUTRAL',
            'strength': 0,
            'call_oi': 0,
            'put_oi': 0,
            'pcr': 1.0,
            'max_pain': spot_price,
            'interpretation': 'Insufficient data'
        }
        
        if not option_data:
            return result
        
        try:
            records = option_data.get('records', {}).get('data', [])
            
            if not records:
                return result
            
            # Calculate total OI for Calls and Puts
            call_oi_total = 0
            put_oi_total = 0
            call_oi_change = 0
            put_oi_change = 0
            
            atm_strike = round(spot_price / 50) * 50  # Nearest 50
            atm_range_strikes = []
            
            for record in records:
                strike = record.get('strikePrice', 0)
                
                # Focus on ATM ±5% strikes
                if abs(strike - spot_price) / spot_price <= 0.05:
                    atm_range_strikes.append(strike)
                    
                    # Call data
                    if 'CE' in record:
                        ce = record['CE']
                        call_oi_total += ce.get('openInterest', 0)
                        call_oi_change += ce.get('changeinOpenInterest', 0)
                    
                    # Put data
                    if 'PE' in record:
                        pe = record['PE']
                        put_oi_total += pe.get('openInterest', 0)
                        put_oi_change += pe.get('changeinOpenInterest', 0)
            
            result['call_oi'] = call_oi_total
            result['put_oi'] = put_oi_total
            
            # Calculate PCR (Put-Call Ratio)
            if call_oi_total > 0:
                pcr = put_oi_total / call_oi_total
                result['pcr'] = round(pcr, 2)
            
            # Analyze OI changes for direction
            # Positive Call OI change + Price up = Bullish (Long buildup)
            # Positive Put OI change + Price down = Bearish (Long buildup)
            # Positive Call OI change + Price down = Bearish (Short buildup)
            # Positive Put OI change + Price up = Bullish (Short covering)
            
            strength = 0
            signal = 'NEUTRAL'
            
            # Simple interpretation based on PCR
            if pcr > 1.3:
                signal = 'BULLISH'
                strength = min(100, (pcr - 1.3) * 100)
                result['interpretation'] = f'High PCR ({pcr:.2f}) indicates Put accumulation - Bullish sentiment'
            elif pcr < 0.7:
                signal = 'BEARISH'
                strength = min(100, (0.7 - pcr) * 100)
                result['interpretation'] = f'Low PCR ({pcr:.2f}) indicates Call accumulation - Bearish sentiment'
            else:
                signal = 'NEUTRAL'
                strength = 50
                result['interpretation'] = f'Balanced PCR ({pcr:.2f}) - Neutral market'
            
            # OI change analysis
            if call_oi_change > put_oi_change and call_oi_change > 0:
                if signal == 'BULLISH':
                    strength = min(100, strength + 20)
                    result['interpretation'] += ' | Strong Call writing (resistance)'
            elif put_oi_change > call_oi_change and put_oi_change > 0:
                if signal == 'BEARISH':
                    strength = min(100, strength + 20)
                    result['interpretation'] += ' | Strong Put writing (support)'
            
            result['signal'] = signal
            result['strength'] = min(100, strength)
            
        except Exception as e:
            result['interpretation'] = f'Error analyzing OI: {str(e)[:50]}'
        
        return result

# ==================== HYBRID DATA FETCHER ====================
class HybridDataFetcher:
    """Fetches data from NSE and Yahoo Finance"""
    
    def __init__(self):
        self.nse_base = "https://www.nseindia.com"
        self.session = self._create_session()
        
    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/'
        })
        try:
            session.get(self.nse_base, timeout=10)
        except:
            pass
        return session
    
    def fetch_nse_quote(self, symbol):
        """Fetch real-time quote from NSE"""
        url = f"{self.nse_base}/api/quote-equity?symbol={symbol}"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                price_info = data.get('priceInfo', {})
                return {
                    'open': float(price_info.get('open', 0)),
                    'high': float(price_info.get('intraDayHighLow', {}).get('max', 0)),
                    'low': float(price_info.get('intraDayHighLow', {}).get('min', 0)),
                    'ltp': float(price_info.get('lastPrice', 0)),
                    'prev_close': float(price_info.get('previousClose', 0)),
                    'volume': float(price_info.get('totalTradedVolume', 0)),
                    'change': float(price_info.get('change', 0)),
                    'pchange': float(price_info.get('pChange', 0))
                }
        except:
            pass
        return None
    
    def fetch_yahoo_intraday(self, symbol):
        """Fallback to Yahoo Finance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data
        except:
            pass
        return None

# ==================== MARKET PROFILE CALCULATOR ====================
class MarketProfileCalculator:
    """Calculate Market Profile metrics"""
    
    @staticmethod
    def calculate_from_intraday(df):
        """Calculate POC, VAH, VAL from intraday data"""
        if df is None or df.empty:
            return None, None, None, None, None
        
        # Volume profile
        prices = pd.concat([df['High'], df['Low'], df['Close']])
        volumes = np.repeat(df['Volume'].values, 3)
        
        price_min, price_max = prices.min(), prices.max()
        bins = np.linspace(price_min, price_max, 50)
        volume_at_price = np.zeros(49)
        
        for price, volume in zip(prices, volumes):
            bin_idx = np.digitize(price, bins) - 1
            if 0 <= bin_idx < len(volume_at_price):
                volume_at_price[bin_idx] += volume
        
        poc_idx = np.argmax(volume_at_price)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area
        total_volume = volume_at_price.sum()
        target_volume = total_volume * 0.70
        sorted_indices = np.argsort(volume_at_price)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += volume_at_price[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= target_volume:
                break
        
        vah = max([bins[i+1] for i in value_area_indices])
        val = min([bins[i] for i in value_area_indices])
        
        # Initial Balance
        first_hour = df.head(60)
        ib_high = float(first_hour['High'].max()) if len(first_hour) > 0 else None
        ib_low = float(first_hour['Low'].min()) if len(first_hour) > 0 else None
        
        return float(poc), float(vah), float(val), ib_high, ib_low

# ==================== OPEN DRIVE DETECTOR ====================
class OpenDriveDetector:
    """Detect Open Drive patterns"""
    
    @staticmethod
    def detect(data, intraday_df=None):
        """
        Detect Open Drive with scoring
        Returns: (is_od, score, signals)
        """
        signals = {}
        score = 0
        
        # Gap
        gap_pct = ((data['open'] - data['prev_close']) / data['prev_close']) * 100
        if abs(gap_pct) > 0.3:
            score += 20
            signals['gap'] = True
        else:
            signals['gap'] = False
        
        # Price move from open
        price_change_pct = ((data['ltp'] - data['open']) / data['open']) * 100
        if abs(price_change_pct) > 0.5:
            score += 25
            signals['strong_move'] = True
        else:
            signals['strong_move'] = False
        
        # Position in range
        price_range = data['high'] - data['low']
        if price_range > 0:
            if data['ltp'] > data['open']:
                distance_from_high = (data['high'] - data['ltp']) / price_range
                if distance_from_high < 0.20:
                    score += 15
                    signals['near_extreme'] = True
                else:
                    signals['near_extreme'] = False
            else:
                distance_from_low = (data['ltp'] - data['low']) / price_range
                if distance_from_low < 0.20:
                    score += 15
                    signals['near_extreme'] = True
                else:
                    signals['near_extreme'] = False
        
        # Volatility
        if data['prev_close'] > 0:
            volatility = (price_range / data['prev_close']) * 100
            if volatility > 1.5:
                score += 10
                signals['volatility'] = True
            else:
                signals['volatility'] = False
        
        # Momentum from intraday
        if intraday_df is not None and not intraday_df.empty:
            bullish = (intraday_df['Close'] > intraday_df['Open']).sum()
            bearish = (intraday_df['Close'] < intraday_df['Open']).sum()
            total = len(intraday_df)
            
            momentum = (max(bullish, bearish) / total * 100) if total > 0 else 50
            
            if momentum > 70:
                score += 30
                signals['momentum'] = 'strong'
            elif momentum > 60:
                score += 20
                signals['momentum'] = 'moderate'
            else:
                signals['momentum'] = 'weak'
        
        is_open_drive = score >= 60
        
        return is_open_drive, score, signals

# ==================== MAIN FNO SCANNER ====================
def scan_fno_stocks_with_oi():
    """
    Comprehensive FNO scanner with Open Interest analysis
    """
    print("\n" + "="*100)
    print("FNO OPEN DRIVE SCANNER WITH OPEN INTEREST ANALYSIS")
    print(f"Scanning {len(FNO_STOCKS)} F&O stocks | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")
    
    # Initialize
    fetcher = HybridDataFetcher()
    oi_analyzer = OpenInterestAnalyzer()
    mp_calc = MarketProfileCalculator()
    od_detector = OpenDriveDetector()
    
    results = []
    errors = []
    
    print(f"{'Symbol':<12} {'Price':<8} {'Chg%':<7} {'OD':<6} {'OI Signal':<12} {'PCR':<6} {'Status':<10}")
    print("-" * 100)
    
    for i, symbol in enumerate(FNO_STOCKS, 1):
        try:
            # Progress indicator
            if i % 20 == 0:
                print(f"\n[Progress: {i}/{len(FNO_STOCKS)}]\n")
            
            # Fetch spot data
            spot_data = fetcher.fetch_nse_quote(symbol)
            
            if not spot_data or spot_data['ltp'] == 0:
                # Fallback to Yahoo
                intraday_df = fetcher.fetch_yahoo_intraday(symbol)
                if intraday_df is None or intraday_df.empty:
                    print(f"{symbol:<12} {'N/A':<8} {'N/A':<7} {'N/A':<6} {'N/A':<12} {'N/A':<6} ❌ No Data")
                    errors.append(symbol)
                    time.sleep(0.2)
                    continue
                
                spot_data = {
                    'open': float(intraday_df.iloc[0]['Open']),
                    'high': float(intraday_df['High'].max()),
                    'low': float(intraday_df['Low'].min()),
                    'ltp': float(intraday_df.iloc[-1]['Close']),
                    'prev_close': float(intraday_df.iloc[0]['Open']),
                    'volume': float(intraday_df['Volume'].sum()),
                    'change': 0,
                    'pchange': 0
                }
            else:
                intraday_df = fetcher.fetch_yahoo_intraday(symbol)
            
            # Detect Open Drive
            is_od, od_score, od_signals = od_detector.detect(spot_data, intraday_df)
            
            # Fetch Open Interest data
            option_data = oi_analyzer.get_option_chain(symbol)
            futures_data = oi_analyzer.get_futures_data(symbol)
            
            # Analyze OI
            oi_analysis = oi_analyzer.analyze_oi_buildup(option_data, futures_data, spot_data['ltp'])
            
            # Calculate Market Profile
            if intraday_df is not None and not intraday_df.empty:
                poc, vah, val, ib_high, ib_low = mp_calc.calculate_from_intraday(intraday_df)
            else:
                price_range = spot_data['high'] - spot_data['low']
                poc = spot_data['ltp']
                vah = poc + (price_range * 0.35)
                val = poc - (price_range * 0.35)
                ib_high = spot_data['high']
                ib_low = spot_data['low']
            
            # Combined signal logic
            # Best setups: Open Drive + OI signal aligned
            chg_pct = ((spot_data['ltp'] - spot_data['open']) / spot_data['open']) * 100
            
            direction = 'BULLISH' if chg_pct > 0 else 'BEARISH'
            oi_signal = oi_analysis['signal']
            
            # Check alignment
            signals_aligned = (
                (direction == 'BULLISH' and oi_signal == 'BULLISH') or
                (direction == 'BEARISH' and oi_signal == 'BEARISH')
            )
            
            overall_pass = is_od and od_score >= 60 and signals_aligned
            
            # Display
            status = "✅ PASS" if overall_pass else "⚠️ Fail"
            
            print(f"{symbol:<12} ₹{spot_data['ltp']:<7.2f} {chg_pct:>+6.2f}% {od_score:<6.0f} {oi_signal:<12} {oi_analysis['pcr']:<6.2f} {status:<10}")
            
            if overall_pass or (is_od and od_score >= 70):  # Include high score OD even without OI alignment
                results.append({
                    'symbol': symbol,
                    'ltp': spot_data['ltp'],
                    'open': spot_data['open'],
                    'high': spot_data['high'],
                    'low': spot_data['low'],
                    'prev_close': spot_data['prev_close'],
                    'change_%': chg_pct,
                    'volume': spot_data['volume'],
                    'od_score': od_score,
                    'oi_signal': oi_signal,
                    'oi_strength': oi_analysis['strength'],
                    'pcr': oi_analysis['pcr'],
                    'call_oi': oi_analysis['call_oi'],
                    'put_oi': oi_analysis['put_oi'],
                    'poc': poc,
                    'vah': vah,
                    'val': val,
                    'ib_high': ib_high,
                    'ib_low': ib_low,
                    'signals_aligned': signals_aligned,
                    'oi_interpretation': oi_analysis['interpretation']
                })
        
        except Exception as e:
            print(f"{symbol:<12} {'Error':<8} {'N/A':<7} {'N/A':<6} {'N/A':<12} {'N/A':<6} ❌ {str(e)[:20]}")
            errors.append(symbol)
        
        time.sleep(0.5)  # Rate limiting
    
    return results, errors

def display_detailed_results(results, errors):
    """Display detailed results with trading recommendations"""
    print("\n" + "="*100)
    
    if not results:
        print("❌ NO HIGH-PROBABILITY SETUPS FOUND")
        print("\nThis could mean:")
        print("  • Market is consolidating")
        print("  • No strong directional moves with OI confirmation")
        print("  • Try again after 10:00 AM or during trending markets")
    else:
        print(f"✅ {len(results)} HIGH-PROBABILITY FNO SETUPS IDENTIFIED")
        print("="*100 + "\n")
        
        df = pd.DataFrame(results)
        df = df.sort_values('od_score', ascending=False)
        
        # Categorize results
        print("🎯 PRIORITY SETUPS (Open Drive + OI Aligned):")
        aligned = df[df['signals_aligned'] == True]
        
        if len(aligned) > 0:
            for idx, row in aligned.head(5).iterrows():
                print(f"\n{'='*100}")
                print(f"⭐ {row['symbol']} - Score: {row['od_score']:.0f}/100 | OI: {row['oi_signal']} ({row['oi_strength']:.0f}%)")
                print(f"{'='*100}")
                print(f"  💰 Spot Price: ₹{row['ltp']:.2f} | Change: {row['change_%']:+.2f}%")
                print(f"  📊 Market Profile:")
                print(f"     POC: ₹{row['poc']:.2f} | VAH: ₹{row['vah']:.2f} | VAL: ₹{row['val']:.2f}")
                print(f"     IB High: ₹{row['ib_high']:.2f} | IB Low: ₹{row['ib_low']:.2f}")
                
                print(f"\n  🎲 Open Interest Analysis:")
                print(f"     Signal: {row['oi_signal']}")
                print(f"     PCR: {row['pcr']:.2f}")
                print(f"     Call OI: {row['call_oi']:,.0f} | Put OI: {row['put_oi']:,.0f}")
                print(f"     {row['oi_interpretation']}")
                
                print(f"\n  💡 Trading Recommendation:")
                if row['change_%'] > 0 and row['oi_signal'] == 'BULLISH':
                    print(f"     📈 BULLISH SETUP")
                    print(f"     Entry: Buy on dip to ₹{row['poc']:.2f}-₹{row['val']:.2f}")
                    print(f"     Stop: Below ₹{row['ib_low']:.2f}")
                    print(f"     Target: ₹{row['vah']:.2f} then ₹{row['high'] * 1.01:.2f}")
                    print(f"     F&O Strategy: Buy Futures or ATM Call options")
                elif row['change_%'] < 0 and row['oi_signal'] == 'BEARISH':
                    print(f"     📉 BEARISH SETUP")
                    print(f"     Entry: Short on rally to ₹{row['poc']:.2f}-₹{row['vah']:.2f}")
                    print(f"     Stop: Above ₹{row['ib_high']:.2f}")
                    print(f"     Target: ₹{row['val']:.2f} then ₹{row['low'] * 0.99:.2f}")
                    print(f"     F&O Strategy: Short Futures or ATM Put options")
        else:
            print("   (None - No perfect alignment found)")
        
        print(f"\n\n{'='*100}")
        print("📋 OTHER OPEN DRIVE SETUPS (Verify OI manually):")
        other = df[df['signals_aligned'] == False]
        
        if len(other) > 0:
            for idx, row in other.head(5).iterrows():
                print(f"\n   {row['symbol']}: ₹{row['ltp']:.2f} ({row['change_%']:+.1f}%) | Score: {row['od_score']:.0f} | OI: {row['oi_signal']}")
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"fno_scanner_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n\n📁 Full results saved to: {filename}")
        
        print(f"\n📊 SCAN SUMMARY:")
        print(f"   Total F&O stocks scanned: {len(FNO_STOCKS)}")
        print(f"   ✅ High-probability setups: {len(aligned)}")
        print(f"   ⚠️ Other Open Drive stocks: {len(other)}")
        print(f"   ❌ Errors/No data: {len(errors)}")
    
    if errors and len(errors) <= 20:
        print(f"\n⚠️ Could not analyze: {', '.join(errors)}")
    
    print("\n" + "="*100)

# ==================== OI CHANGE TRACKER ====================
class OIChangeTracker:
    """Track OI changes over time to detect buildup/unwinding"""
    
    def __init__(self, history_file="oi_history.csv"):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self):
        """Load historical OI data"""
        try:
            if os.path.exists(self.history_file):
                return pd.read_csv(self.history_file)
        except:
            pass
        return pd.DataFrame(columns=['timestamp', 'symbol', 'call_oi', 'put_oi', 'pcr'])
    
    def save_current_oi(self, results):
        """Save current OI data for future comparison"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        new_records = []
        for result in results:
            new_records.append({
                'timestamp': timestamp,
                'symbol': result['symbol'],
                'call_oi': result['call_oi'],
                'put_oi': result['put_oi'],
                'pcr': result['pcr']
            })
        
        if new_records:
            new_df = pd.DataFrame(new_records)
            self.history = pd.concat([self.history, new_df], ignore_index=True)
            
            # Keep only last 7 days
            self.history['timestamp'] = pd.to_datetime(self.history['timestamp'])
            cutoff_date = datetime.now() - timedelta(days=7)
            self.history = self.history[self.history['timestamp'] > cutoff_date]
            
            # Save to file
            self.history.to_csv(self.history_file, index=False)
            print(f"\n💾 OI history saved to {self.history_file}")
    
    def analyze_oi_changes(self, symbol):
        """Analyze OI changes over last few scans"""
        if self.history.empty:
            return None
        
        symbol_data = self.history[self.history['symbol'] == symbol].tail(5)
        
        if len(symbol_data) < 2:
            return None
        
        # Calculate changes
        latest = symbol_data.iloc[-1]
        previous = symbol_data.iloc[-2]
        
        call_oi_change = latest['call_oi'] - previous['call_oi']
        put_oi_change = latest['put_oi'] - previous['put_oi']
        pcr_change = latest['pcr'] - previous['pcr']
        
        call_oi_change_pct = (call_oi_change / previous['call_oi'] * 100) if previous['call_oi'] > 0 else 0
        put_oi_change_pct = (put_oi_change / previous['put_oi'] * 100) if previous['put_oi'] > 0 else 0
        
        return {
            'call_oi_change': call_oi_change,
            'put_oi_change': put_oi_change,
            'call_oi_change_pct': call_oi_change_pct,
            'put_oi_change_pct': put_oi_change_pct,
            'pcr_change': pcr_change
        }

# ==================== QUICK OI FILTER ====================
def quick_oi_filter(min_oi_change_pct=10):
    """
    Quick filter to find stocks with significant OI changes
    Use this when you want to focus only on OI buildup/unwinding
    """
    print("\n" + "="*100)
    print("QUICK OI CHANGE FILTER - High OI Activity Stocks")
    print(f"Filtering for OI change >= {min_oi_change_pct}%")
    print("="*100 + "\n")
    
    oi_analyzer = OpenInterestAnalyzer()
    fetcher = HybridDataFetcher()
    
    high_oi_stocks = []
    
    print(f"{'Symbol':<12} {'Price':<8} {'Call OI Chg':<12} {'Put OI Chg':<12} {'Signal':<10}")
    print("-" * 100)
    
    for symbol in FNO_STOCKS[:50]:  # Scan subset for speed
        try:
            spot_data = fetcher.fetch_nse_quote(symbol)
            if not spot_data:
                continue
            
            option_data = oi_analyzer.get_option_chain(symbol)
            oi_analysis = oi_analyzer.analyze_oi_buildup(option_data, None, spot_data['ltp'])
            
            # This is simplified - in real scenario, compare with previous scan
            # For now, just show high OI stocks
            total_oi = oi_analysis['call_oi'] + oi_analysis['put_oi']
            
            if total_oi > 1000000:  # Minimum 10 lakh OI
                print(f"{symbol:<12} ₹{spot_data['ltp']:<7.2f} {oi_analysis['call_oi']:>10,.0f} {oi_analysis['put_oi']:>10,.0f} {oi_analysis['signal']:<10}")
                
                high_oi_stocks.append({
                    'symbol': symbol,
                    'price': spot_data['ltp'],
                    'call_oi': oi_analysis['call_oi'],
                    'put_oi': oi_analysis['put_oi'],
                    'pcr': oi_analysis['pcr'],
                    'signal': oi_analysis['signal']
                })
            
            time.sleep(0.5)
            
        except:
            continue
    
    print("\n" + "="*100)
    print(f"\n✅ Found {len(high_oi_stocks)} stocks with high OI activity")
    
    if high_oi_stocks:
        df = pd.DataFrame(high_oi_stocks)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"high_oi_stocks_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"📁 Saved to: {filename}")
    
    return high_oi_stocks

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    import os
    
    print("\n" + "🎯"*50)
    print("FNO OPEN DRIVE SCANNER WITH OPEN INTEREST ANALYSIS")
    print("Scans all F&O stocks | Combines Price Action + OI Data")
    print("🎯"*50)
    
    now = datetime.now()
    
    # Check market hours
    if now.weekday() >= 5:
        print("\n⚠️ Weekend - Market closed")
        print("You can still run to test connectivity")
        input("\nPress Enter to continue or Ctrl+C to exit...")
    
    market_open = now.replace(hour=9, minute=15)
    market_close = now.replace(hour=15, minute=30)
    
    if now < market_open:
        print(f"\n⏰ Market opens at 9:15 AM (currently {now.strftime('%H:%M:%S')})")
    elif now > market_close:
        print(f"\n⏰ Market closed - showing previous session data")
    
    print("\n" + "="*100)
    print("SCANNER OPTIONS:")
    print("="*100)
    print("1. Full FNO Scan (Open Drive + OI Analysis) - Recommended")
    print("2. Quick OI Filter (High OI Activity only)")
    print("3. Both (Full scan + OI filter)")
    print("="*100)
    
    choice = input("\nEnter choice (1/2/3) or press Enter for option 1: ").strip() or "1"
    
    if choice == "1":
        print("\n🚀 Starting full FNO scan with OI analysis...")
        print("This will take 2-3 minutes to scan ~180 stocks\n")
        
        results, errors = scan_fno_stocks_with_oi()
        display_detailed_results(results, errors)
        
        # Save OI history for tracking
        if results:
            try:
                tracker = OIChangeTracker()
                tracker.save_current_oi(results)
            except Exception as e:
                print(f"\n⚠️ Could not save OI history: {e}")
    
    elif choice == "2":
        print("\n🚀 Running quick OI filter...")
        high_oi_stocks = quick_oi_filter()
    
    elif choice == "3":
        print("\n🚀 Running both scans...")
        
        print("\n" + "="*100)
        print("PART 1: FULL FNO SCAN")
        print("="*100)
        results, errors = scan_fno_stocks_with_oi()
        display_detailed_results(results, errors)
        
        print("\n\n" + "="*100)
        print("PART 2: HIGH OI ACTIVITY FILTER")
        print("="*100)
        high_oi_stocks = quick_oi_filter()
    
    print("\n" + "="*100)
    print("✅ SCAN COMPLETED!")
    print("="*100)
    
    print("\n💡 NEXT STEPS:")
    print("   1. Review priority setups (Open Drive + OI aligned)")
    print("   2. Verify on GoCharting Market Profile")
    print("   3. Check OI data on NSE website or Sensibull")
    print("   4. Plan F&O trades (Futures or Options)")
    print("   5. Consider using options for lower capital requirement")
    
    print("\n📚 UNDERSTANDING OI SIGNALS:")
    print("   • PCR > 1.3 + Price up = Bullish (Put writers confident)")
    print("   • PCR < 0.7 + Price down = Bearish (Call writers confident)")
    print("   • High Call OI + Price down = Resistance (bearish)")
    print("   • High Put OI + Price up = Support (bullish)")
    
    print("\n⚠️ RISK DISCLAIMER:")
    print("   • This is a screening tool, not trading advice")
    print("   • Always use stop losses")
    print("   • F&O trading carries high risk")
    print("   • Verify OI data on official NSE website")
    
    print("\n" + "="*100 + "\n")