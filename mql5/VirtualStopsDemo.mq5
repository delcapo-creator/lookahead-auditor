//+------------------------------------------------------------------+
//|                                             VirtualStopsDemo.mq5 |
//|  A zero-edge EA that manages VIRTUAL stop-loss / take-profit on   |
//|  every new bar, the way many EAs with hidden or virtual stops do. |
//|                                                                  |
//|  InpApplyGuard = false : the position is opened at the close of  |
//|                  bar 1 and then checked against the High/Low of  |
//|                  that SAME bar -> entry-bar look-ahead.          |
//|  InpApplyGuard = true  : exits are checked only on bars that     |
//|                  start after the entry -> correct simulation.    |
//|                                                                  |
//|  Direction is a coin flip, so the true edge is zero. Any profit  |
//|  reported with the guard OFF is produced by the bug alone.       |
//+------------------------------------------------------------------+
#property copyright "Saad"
#property version   "1.00"

input bool   InpApplyGuard = true;   // Apply the entry-bar look-ahead guard
input double InpSlAtr      = 1.0;    // Stop-loss distance, x ATR
input double InpTpAtr      = 0.5;    // Take-profit distance, x ATR
input double InpEntryProb  = 0.15;   // Probability of opening on a new bar
input int    InpAtrPeriod  = 14;     // ATR period
input int    InpSeed       = 7;      // Random seed (same seed = same coin flips)

int      g_atr      = INVALID_HANDLE;
datetime g_last_bar = 0;

//--- virtual position
int      g_dir       = 0;            // +1 long, -1 short, 0 flat
double   g_entry     = 0.0;
double   g_sl        = 0.0;
double   g_tp        = 0.0;
datetime g_entry_bar = 0;            // open time of the bar whose close was the entry

//--- statistics, in price units
int      g_trades = 0, g_wins = 0, g_same_bar = 0;
double   g_gp = 0.0, g_gl = 0.0, g_sum = 0.0, g_sumsq = 0.0, g_sum_bars = 0.0;

//+------------------------------------------------------------------+
double Rand01()
  {
   return MathRand() / 32768.0;
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   g_atr = iATR(_Symbol, _Period, InpAtrPeriod);
   if(g_atr == INVALID_HANDLE)
      return INIT_FAILED;
   MathSrand(InpSeed);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void CloseVirtual(const double price, const datetime exit_bar)
  {
   double pnl  = (price - g_entry) * g_dir;
   int    bars = iBarShift(_Symbol, _Period, g_entry_bar, false)
                 - iBarShift(_Symbol, _Period, exit_bar, false);

   g_trades++;
   g_sum      += pnl;
   g_sumsq    += pnl * pnl;
   g_sum_bars += bars;
   if(bars == 0)
      g_same_bar++;
   if(pnl > 0.0)
     {
      g_wins++;
      g_gp += pnl;
     }
   else
      g_gl -= pnl;

   g_dir = 0;
  }

//+------------------------------------------------------------------+
//| Pessimistic exit check on one closed bar: gaps fill at the open, |
//| and the stop-loss wins when SL and TP are both inside the bar.   |
//+------------------------------------------------------------------+
void CheckExits(const datetime t, const double o, const double h, const double l)
  {
   bool   gap_ok = (t != g_entry_bar);   // no gap on the bar we entered at its close
   bool   hit    = false;
   double px     = 0.0;

   if(g_dir == 1)
     {
      if(gap_ok && o <= g_sl)      { px = o;    hit = true; }
      else if(gap_ok && o >= g_tp) { px = g_tp; hit = true; }
      else if(l <= g_sl)           { px = g_sl; hit = true; }
      else if(h >= g_tp)           { px = g_tp; hit = true; }
     }
   else
      if(g_dir == -1)
        {
         if(gap_ok && o >= g_sl)      { px = o;    hit = true; }
         else if(gap_ok && o <= g_tp) { px = g_tp; hit = true; }
         else if(h >= g_sl)           { px = g_sl; hit = true; }
         else if(l <= g_tp)           { px = g_tp; hit = true; }
        }

   if(hit)
      CloseVirtual(px, t);
  }

//+------------------------------------------------------------------+
void TryOpen(const datetime t, const double close_price, const double atr,
             const double r_entry, const double r_dir)
  {
   if(r_entry >= InpEntryProb)
      return;
   g_dir       = (r_dir < 0.5) ? 1 : -1;
   g_entry     = close_price;
   g_entry_bar = t;
   g_sl        = g_entry - g_dir * InpSlAtr * atr;
   g_tp        = g_entry + g_dir * InpTpAtr * atr;
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   datetime t0 = iTime(_Symbol, _Period, 0);
   if(t0 == g_last_bar)
      return;                              // act once per new bar
   g_last_bar = t0;

   //--- draw the coin flips on every bar, so both modes see the same sequence
   double r_entry = Rand01();
   double r_dir   = Rand01();

   if(Bars(_Symbol, _Period) < InpAtrPeriod + 3)
      return;
   double atr[1];
   if(CopyBuffer(g_atr, 0, 1, 1, atr) != 1)
      return;

   datetime t1 = iTime(_Symbol, _Period, 1);
   double   o1 = iOpen(_Symbol, _Period, 1);
   double   h1 = iHigh(_Symbol, _Period, 1);
   double   l1 = iLow(_Symbol, _Period, 1);
   double   c1 = iClose(_Symbol, _Period, 1);

   if(InpApplyGuard)
     {
      //--- correct order: manage the open position on the bar that just
      //--- closed, then open a new one at that bar's close
      if(g_dir != 0)
         CheckExits(t1, o1, h1, l1);
      if(g_dir == 0)
         TryOpen(t1, c1, atr[0], r_entry, r_dir);
     }
   else
     {
      //--- leaky order: open at the close of bar 1, then check the new
      //--- position against the High/Low of that SAME bar
      if(g_dir == 0)
         TryOpen(t1, c1, atr[0], r_entry, r_dir);
      if(g_dir != 0)
         CheckExits(t1, o1, h1, l1);
     }
  }

//+------------------------------------------------------------------+
double ProfitFactor()
  {
   return (g_gl > 0.0) ? g_gp / g_gl : 0.0;
  }

//+------------------------------------------------------------------+
double OnTester()
  {
   return ProfitFactor();
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_atr != INVALID_HANDLE)
      IndicatorRelease(g_atr);
   if(g_trades < 2)
     {
      Print("VirtualStopsDemo: not enough virtual trades.");
      return;
     }
   double mean  = g_sum / g_trades;
   double var   = (g_sumsq - g_trades * mean * mean) / (g_trades - 1);
   double tstat = (var > 0.0) ? mean / MathSqrt(var) * MathSqrt(g_trades) : 0.0;

   PrintFormat("=== VirtualStopsDemo: look-ahead guard %s ===", InpApplyGuard ? "ON" : "OFF");
   PrintFormat("Virtual trades        : %d", g_trades);
   PrintFormat("Win rate              : %.1f%%", 100.0 * g_wins / g_trades);
   PrintFormat("Profit factor         : %.2f", ProfitFactor());
   PrintFormat("t-stat of mean trade  : %.2f", tstat);
   PrintFormat("Average bars in trade : %.2f", g_sum_bars / g_trades);
   PrintFormat("Closed in entry bar   : %.1f%%", 100.0 * g_same_bar / g_trades);
  }
//+------------------------------------------------------------------+
