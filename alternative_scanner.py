"""
Alternative Open Drive Scanner - Different Algorithm & Approach
Uses hybrid data sources and refined detection logic
Can run alongside the primary scanner for confirmation
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# ==================== MULTI-SOURCE DATA FETCHER ====================
class HybridDataFetcher:
    """Fetches data from multiple sources for reliability"""
    
    def __init__(self):
        self.nse_base = "https://www.nseindia.com"
        self.session = self._create_session()
        
    def _create_session(self):
        """Create session with rotating headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/'
        })
        # Get cookies
        try:
            session.get(self.nse_base, timeout=10)
        except:
            pass
        return session
    
    def fetch_nse_indices_data(self):
        """Fetch all Nifty 50 stocks in one call"""
        url = f"{self.nse_base}/api/equity-stockIndices?index=NIFTY%2050"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                stocks = {}
                for item in data.get('data', []):
                    stocks[item['symbol']] = {
                        'open': float(item.get('open', 0)),
                        'high': float(item.get('dayHigh', 0)),
                        'low': float(item.get('dayLow', 0)),
                        'ltp': float(item.get('lastPrice', 0)),
                        'prev_close': float(item.get('previousClose', 0)),
                        'volume': float(item.get('totalTradedVolume', 0)),
                        'change': float(item.get('pChange', 0)),
                        'timestamp': datetime.now()
                    }
                return stocks
        except Exception as e:
            print(f"NSE fetch error: {e}")
        return None
    
    def fetch_yahoo_intraday(self, symbol):
        """Fallback to Yahoo Finance for detailed data"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data
        except:
            pass
        return None

# ==================== ALTERNATIVE MARKET PROFILE ALGORITHM ====================
class EnhancedMarketProfile:
    """
    Advanced Market Profile calculator
    Uses volume-weighted price distribution
    """
    
    @staticmethod
    def calculate_volume_profile(prices, volumes, num_bins=50):
        """
        Calculate volume profile using histogram approach
        Returns price levels and their volume distribution
        """
        if len(prices) == 0 or len(volumes) == 0:
            return None, None, None
        
        price_min, price_max = prices.min(), prices.max()
        
        # Create price bins
        bins = np.linspace(price_min, price_max, num_bins)
        volume_at_price = np.zeros(num_bins - 1)
        
        # Distribute volume across price levels
        for price, volume in zip(prices, volumes):
            bin_idx = np.digitize(price, bins) - 1
            if 0 <= bin_idx < len(volume_at_price):
                volume_at_price[bin_idx] += volume
        
        # Find POC (Point of Control)
        poc_idx = np.argmax(volume_at_price)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Calculate Value Area (70% of volume)
        total_volume = volume_at_price.sum()
        target_volume = total_volume * 0.70
        
        # Sort by volume and accumulate
        sorted_indices = np.argsort(volume_at_price)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += volume_at_price[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= target_volume:
                break
        
        # Get VAH and VAL
        vah = max([bins[i+1] for i in value_area_indices])
        val = min([bins[i] for i in value_area_indices])
        
        return poc, vah, val
    
    @staticmethod
    def calculate_initial_balance_accurate(df):
        """
        Calculate IB from actual minute data
        IB = first 60 minutes (9:15 to 10:15)
        """
        if df is None or df.empty:
            return None, None
        
        # Get first 60 data points
        ib_data = df.head(60)
        
        if len(ib_data) < 10:  # Need at least 10 minutes
            return None, None
        
        ib_high = ib_data['High'].max()
        ib_low = ib_data['Low'].min()
        
        return float(ib_high), float(ib_low)

# ==================== ENHANCED OPEN DRIVE DETECTOR ====================
class OpenDriveDetector:
    """
    Advanced Open Drive detection with multiple confirmation signals
    """
    
    @staticmethod
    def calculate_momentum_score(intraday_df):
        """Calculate momentum strength from price action"""
        if intraday_df is None or intraday_df.empty or len(intraday_df) < 5:
            return 0
        
        # Calculate consecutive moves in same direction
        closes = intraday_df['Close'].values
        changes = np.diff(closes)
        
        # Count consecutive positive/negative moves
        current_streak = 0
        max_streak = 0
        current_sign = 0
        
        for change in changes:
            sign = 1 if change > 0 else -1 if change < 0 else 0
            if sign == current_sign and sign != 0:
                current_streak += 1
            else:
                max_streak = max(max_streak, abs(current_streak))
                current_streak = 1
                current_sign = sign
        
        max_streak = max(max_streak, abs(current_streak))
        
        # Momentum score (0-100)
        momentum = min(100, (max_streak / len(changes)) * 100 * 2)
        return momentum
    
    @staticmethod
    def detect_open_drive(data, intraday_df=None):
        """
        Multi-factor Open Drive detection
        Returns: (is_open_drive, score, signals)
        """
        signals = {}
        score = 0
        
        # Factor 1: Gap (20 points)
        gap_pct = ((data['open'] - data['prev_close']) / data['prev_close']) * 100
        if abs(gap_pct) > 0.3:
            score += 20
            signals['gap'] = True
        else:
            signals['gap'] = False
        
        # Factor 2: Price change from open (25 points)
        price_change_pct = ((data['ltp'] - data['open']) / data['open']) * 100
        if abs(price_change_pct) > 0.5:
            score += 25
            signals['strong_move'] = True
        elif abs(price_change_pct) > 0.3:
            score += 15
            signals['strong_move'] = 'moderate'
        else:
            signals['strong_move'] = False
        
        # Factor 3: Price near extreme (15 points)
        price_range = data['high'] - data['low']
        if price_range > 0:
            if data['ltp'] > data['open']:
                # Bullish - should be near high
                distance_from_high = (data['high'] - data['ltp']) / price_range
                if distance_from_high < 0.15:  # Within 15% of high
                    score += 15
                    signals['near_extreme'] = True
                elif distance_from_high < 0.30:
                    score += 8
                    signals['near_extreme'] = 'moderate'
                else:
                    signals['near_extreme'] = False
            else:
                # Bearish - should be near low
                distance_from_low = (data['ltp'] - data['low']) / price_range
                if distance_from_low < 0.15:
                    score += 15
                    signals['near_extreme'] = True
                elif distance_from_low < 0.30:
                    score += 8
                    signals['near_extreme'] = 'moderate'
                else:
                    signals['near_extreme'] = False
        
        # Factor 4: Volatility (10 points)
        if data['prev_close'] > 0:
            volatility = (price_range / data['prev_close']) * 100
            if volatility > 1.5:
                score += 10
                signals['volatility'] = True
            elif volatility > 1.0:
                score += 5
                signals['volatility'] = 'moderate'
            else:
                signals['volatility'] = False
        
        # Factor 5: Momentum from intraday (30 points)
        if intraday_df is not None and not intraday_df.empty:
            momentum = OpenDriveDetector.calculate_momentum_score(intraday_df)
            if momentum > 70:
                score += 30
                signals['momentum'] = 'strong'
            elif momentum > 50:
                score += 20
                signals['momentum'] = 'moderate'
            elif momentum > 30:
                score += 10
                signals['momentum'] = 'weak'
            else:
                signals['momentum'] = False
        
        # Determine if it's an Open Drive (score >= 60)
        is_open_drive = score >= 60
        
        return is_open_drive, score, signals

# ==================== VALUE AREA COMPARATOR ====================
class ValueAreaAnalyzer:
    """Analyzes Value Area migration"""
    
    @staticmethod
    def analyze_va_shift(current_data, prev_data, current_poc, current_vah, current_val):
        """
        Analyze if Value Area has shifted bullish
        Returns: (is_bullish, confidence, details)
        """
        details = {}
        confidence = 0
        
        # Estimate previous day's Value Area
        prev_range = prev_data['high'] - prev_data['low']
        prev_poc = prev_data['close']
        prev_vah = prev_poc + (prev_range * 0.35)
        prev_val = prev_poc - (prev_range * 0.35)
        
        # Check 1: VAH migration (40% weight)
        vah_shift = current_vah - prev_vah
        vah_shift_pct = (vah_shift / prev_vah) * 100
        
        if vah_shift_pct > 0.5:
            confidence += 40
            details['vah_shift'] = 'strong'
        elif vah_shift_pct > 0.2:
            confidence += 25
            details['vah_shift'] = 'moderate'
        else:
            details['vah_shift'] = 'weak'
        
        # Check 2: POC migration (30% weight)
        poc_shift = current_poc - prev_poc
        poc_shift_pct = (poc_shift / prev_poc) * 100
        
        if poc_shift_pct > 0.3:
            confidence += 30
            details['poc_shift'] = 'strong'
        elif poc_shift_pct > 0.1:
            confidence += 20
            details['poc_shift'] = 'moderate'
        else:
            details['poc_shift'] = 'weak'
        
        # Check 3: VAL support (20% weight)
        val_shift = current_val - prev_val
        if val_shift >= 0:  # VAL didn't drop
            confidence += 20
            details['val_support'] = True
        elif val_shift >= -0.005 * prev_val:  # Within 0.5%
            confidence += 10
            details['val_support'] = 'marginal'
        else:
            details['val_support'] = False
        
        # Check 4: Price acceptance (10% weight)
        if current_data['ltp'] > prev_vah:
            confidence += 10
            details['price_acceptance'] = 'above_prev_vah'
        elif current_data['ltp'] > prev_poc:
            confidence += 5
            details['price_acceptance'] = 'above_prev_poc'
        else:
            details['price_acceptance'] = 'below_prev_va'
        
        is_bullish = confidence >= 60
        
        details['prev_vah'] = prev_vah
        details['prev_val'] = prev_val
        details['prev_poc'] = prev_poc
        
        return is_bullish, confidence, details

# ==================== NIFTY 50 STOCKS ====================
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
    "BHARTIARTL", "SBIN", "KOTAKBANK", "ITC", "LT", "AXISBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH", "SUNPHARMA",
    "TITAN", "ULTRACEMCO", "NESTLEIND", "WIPRO", "ONGC", "NTPC",
    "POWERGRID", "TATAMOTORS", "BAJAJFINSV", "TATASTEEL", "M&M",
    "TECHM", "ADANIPORTS", "COALINDIA", "JSWSTEEL", "INDUSINDBK",
    "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT", "HINDALCO",
    "BRITANNIA", "GRASIM", "APOLLOHOSP", "BPCL", "HEROMOTOCO",
    "TATACONSUM", "BAJAJ-AUTO", "SBILIFE", "HDFCLIFE", "SHRIRAMFIN"
]

# ==================== MAIN SCANNER ====================
def scan_open_drive_stocks():
    """
    Alternative Open Drive scanner with scoring system
    """
    print("\n" + "="*90)
    print("ALTERNATIVE OPEN DRIVE SCANNER - MULTI-FACTOR SCORING SYSTEM")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("="*90 + "\n")
    
    # Initialize
    fetcher = HybridDataFetcher()
    mp_calc = EnhancedMarketProfile()
    od_detector = OpenDriveDetector()
    va_analyzer = ValueAreaAnalyzer()
    
    # Fetch all Nifty 50 data
    print("📊 Fetching live market data from NSE...")
    nse_data = fetcher.fetch_nse_indices_data()
    
    if not nse_data:
        print("⚠️ NSE data unavailable, falling back to Yahoo Finance...")
        use_yahoo = True
    else:
        print(f"✅ Fetched data for {len(nse_data)} stocks\n")
        use_yahoo = False
    
    results = []
    errors = []
    
    print(f"{'Symbol':<12} {'Price':<8} {'Chg%':<7} {'OD Score':<9} {'VA Score':<9} {'Status':<10}")
    print("-" * 90)
    
    for symbol in NIFTY_50_SYMBOLS:
        try:
            # Get current data
            if not use_yahoo and symbol in nse_data:
                current_data = nse_data[symbol]
                intraday_df = fetcher.fetch_yahoo_intraday(symbol)
            else:
                # Fallback to Yahoo
                intraday_df = fetcher.fetch_yahoo_intraday(symbol)
                if intraday_df is None or intraday_df.empty:
                    print(f"{symbol:<12} {'N/A':<8} {'N/A':<7} {'N/A':<9} {'N/A':<9} ❌ No Data")
                    errors.append(symbol)
                    continue
                
                current_data = {
                    'open': float(intraday_df.iloc[0]['Open']),
                    'high': float(intraday_df['High'].max()),
                    'low': float(intraday_df['Low'].min()),
                    'ltp': float(intraday_df.iloc[-1]['Close']),
                    'volume': float(intraday_df['Volume'].sum()),
                    'prev_close': float(intraday_df.iloc[0]['Open'])  # Approximation
                }
            
            # Detect Open Drive
            is_od, od_score, od_signals = od_detector.detect_open_drive(current_data, intraday_df)
            
            # Calculate Market Profile
            if intraday_df is not None and not intraday_df.empty:
                prices = pd.concat([intraday_df['High'], intraday_df['Low'], intraday_df['Close']])
                volumes = np.repeat(intraday_df['Volume'].values, 3)
                poc, vah, val = mp_calc.calculate_volume_profile(prices.values, volumes)
                ib_high, ib_low = mp_calc.calculate_initial_balance_accurate(intraday_df)
            else:
                # Fallback estimation
                price_range = current_data['high'] - current_data['low']
                poc = current_data['ltp']
                vah = poc + (price_range * 0.35)
                val = poc - (price_range * 0.35)
                ib_high = current_data['high']
                ib_low = current_data['low']
            
            # Analyze Value Area
            prev_data = {
                'high': current_data['prev_close'] * 1.015,
                'low': current_data['prev_close'] * 0.985,
                'close': current_data['prev_close']
            }
            
            is_va_bullish, va_confidence, va_details = va_analyzer.analyze_va_shift(
                current_data, prev_data, poc, vah, val
            )
            
            # Calculate overall score
            overall_pass = is_od and is_va_bullish and od_score >= 60 and va_confidence >= 60
            
            # Display
            chg_pct = ((current_data['ltp'] - current_data['open']) / current_data['open']) * 100
            status = "✅ PASS" if overall_pass else "⚠️ Fail"
            
            print(f"{symbol:<12} ₹{current_data['ltp']:<7.2f} {chg_pct:>+6.2f}% {od_score:<9.0f} {va_confidence:<9.0f} {status:<10}")
            
            if overall_pass:
                results.append({
                    'symbol': symbol,
                    'ltp': current_data['ltp'],
                    'open': current_data['open'],
                    'high': current_data['high'],
                    'low': current_data['low'],
                    'change_%': chg_pct,
                    'prev_close': current_data['prev_close'],
                    'od_score': od_score,
                    'va_confidence': va_confidence,
                    'poc': poc,
                    'vah': vah,
                    'val': val,
                    'ib_high': ib_high,
                    'ib_low': ib_low,
                    'momentum': od_signals.get('momentum', 'N/A'),
                    'gap': od_signals.get('gap', False),
                    'va_shift': va_details.get('vah_shift', 'N/A')
                })
        
        except Exception as e:
            print(f"{symbol:<12} {'Error':<8} {'N/A':<7} {'N/A':<9} {'N/A':<9} ❌ {str(e)[:20]}")
            errors.append(symbol)
        
        time.sleep(0.3)
    
    return results, errors

def display_detailed_results(results, errors):
    """Display detailed results with trading insights"""
    print("\n" + "="*90)
    
    if not results:
        print("❌ NO OPEN DRIVE STOCKS FOUND")
        print("\nPossible reasons:")
        print("  • Market is ranging without strong directional bias")
        print("  • No stocks met the multi-factor scoring threshold (60+)")
        print("  • Run again after 10:00 AM for more data")
    else:
        print(f"✅ {len(results)} OPEN DRIVE CANDIDATES IDENTIFIED")
        print("="*90 + "\n")
        
        df = pd.DataFrame(results)
        df = df.sort_values('od_score', ascending=False)
        
        # Detailed view
        for idx, row in df.iterrows():
            print(f"\n{'='*90}")
            print(f"🎯 {row['symbol']} - Open Drive Score: {row['od_score']:.0f}/100 | VA Score: {row['va_confidence']:.0f}/100")
            print(f"{'='*90}")
            print(f"  💰 Price Action:")
            print(f"     Current: ₹{row['ltp']:.2f} | Open: ₹{row['open']:.2f} | Change: {row['change_%']:+.2f}%")
            print(f"     High: ₹{row['high']:.2f} | Low: ₹{row['low']:.2f}")
            
            print(f"\n  📊 Market Profile:")
            print(f"     POC (Point of Control): ₹{row['poc']:.2f}")
            print(f"     VAH (Value Area High):  ₹{row['vah']:.2f}")
            print(f"     VAL (Value Area Low):   ₹{row['val']:.2f}")
            
            print(f"\n  🎯 Initial Balance (First Hour):")
            print(f"     IB High: ₹{row['ib_high']:.2f}")
            print(f"     IB Low:  ₹{row['ib_low']:.2f}")
            print(f"     IB Range: ₹{row['ib_high'] - row['ib_low']:.2f}")
            
            print(f"\n  🔍 Signals:")
            print(f"     Momentum: {row['momentum']}")
            print(f"     Gap: {'Yes' if row['gap'] else 'No'}")
            print(f"     VA Shift: {row['va_shift']}")
            
            print(f"\n  💡 Trading Plan:")
            if row['change_%'] > 0:
                print(f"     Direction: BULLISH 📈")
                print(f"     Entry: Pullback to POC (₹{row['poc']:.2f}) or VAL (₹{row['val']:.2f})")
                print(f"     Stop Loss: Below ₹{row['ib_low']:.2f}")
                print(f"     Target 1: ₹{row['vah']:.2f}")
                print(f"     Target 2: ₹{row['high'] + (row['high'] - row['ib_low']):.2f}")
            else:
                print(f"     Direction: BEARISH 📉")
                print(f"     Entry: Pullback to POC (₹{row['poc']:.2f}) or VAH (₹{row['vah']:.2f})")
                print(f"     Stop Loss: Above ₹{row['ib_high']:.2f}")
                print(f"     Target 1: ₹{row['val']:.2f}")
                print(f"     Target 2: ₹{row['low'] - (row['ib_high'] - row['low']):.2f}")
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"alternative_scanner_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n\n📁 Results saved to: {filename}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Scanned: {len(NIFTY_50_SYMBOLS)}")
        print(f"   ✅ Passed: {len(results)}")
        print(f"   ⚠️ Failed: {len(NIFTY_50_SYMBOLS) - len(results) - len(errors)}")
        print(f"   ❌ Errors: {len(errors)}")
        
        if len(results) > 0:
            print(f"\n🏆 TOP PICK: {df.iloc[0]['symbol']} (Score: {df.iloc[0]['od_score']:.0f})")
    
    if errors:
        print(f"\n⚠️ Could not analyze: {', '.join(errors[:10])}")
    
    print("\n" + "="*90)

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "🔍"*45)
    print("ALTERNATIVE OPEN DRIVE SCANNER")
    print("Multi-Factor Scoring | Hybrid Data Sources | Enhanced Detection")
    print("🔍"*45)
    
    now = datetime.now()
    
    if now.weekday() >= 5:
        print("\n⚠️ Weekend - Market closed")
        input("Press Enter to run with test data or Ctrl+C to exit...")
    
    print("\n🚀 Starting comprehensive scan...")
    print("This scanner uses a different algorithm for comparison/validation\n")
    
    results, errors = scan_open_drive_stocks()
    display_detailed_results(results, errors)
    
    print("\n✅ Alternative scan completed!")
    print("\n💡 RECOMMENDATION:")
    print("   • Compare results with primary scanner for confirmation")
    print("   • Stocks appearing in BOTH scanners = highest confidence")
    print("   • Always verify on GoCharting before trading")
    print("\n" + "="*90 + "\n")