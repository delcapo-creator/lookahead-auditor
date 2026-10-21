//+------------------------------------------------------------------+
//|                                             SameBarExitCheck.mqh |
//|  Pairs entry/exit deals by position ID and reports how many      |
//|  positions closed inside the bar they were opened on.            |
//|  Works on account history (from a script) and on Strategy Tester |
//|  history (call it from an EA's OnTester or OnDeinit).            |
//+------------------------------------------------------------------+
#property copyright "Saad"

struct EntryRec
  {
   ulong             pos_id;
   datetime          time;
   string            symbol;
   double            commission;
  };

EntryRec g_entries[];

//+------------------------------------------------------------------+
int FindEntry(const ulong pos_id)
  {
   for(int i = ArraySize(g_entries) - 1; i >= 0; i--)
      if(g_entries[i].pos_id == pos_id)
         return i;
   return -1;
  }

//+------------------------------------------------------------------+
double ProfitFactor(const double gross_profit, const double gross_loss)
  {
   if(gross_loss > 0.0)
      return gross_profit / gross_loss;
   return (gross_profit > 0.0) ? DBL_MAX : 0.0;
  }

//+------------------------------------------------------------------+
string PfToString(const double pf)
  {
   return (pf == DBL_MAX) ? "inf" : DoubleToString(pf, 2);
  }

//+------------------------------------------------------------------+
bool SameBarExitReport(const ENUM_TIMEFRAMES tf, const datetime from, const datetime to,
                       const string symbol_filter = "", const long magic_filter = -1)
  {
   ArrayResize(g_entries, 0);
   if(!HistorySelect(from, to))
     {
      Print("HistorySelect failed, error ", GetLastError());
      return false;
     }

   int    total     = HistoryDealsTotal();
   int    n_all     = 0, n_same = 0, n_skipped = 0;
   double sum_bars  = 0.0;
   double gp_all    = 0.0, gl_all = 0.0;   // every closing deal
   double gp_later  = 0.0, gl_later = 0.0; // closing deals after the entry bar

   for(int i = 0; i < total; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;

      long type = HistoryDealGetInteger(ticket, DEAL_TYPE);
      if(type != DEAL_TYPE_BUY && type != DEAL_TYPE_SELL)
         continue;                                   // skip balance, credit, etc.

      string sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
      if(symbol_filter != "" && sym != symbol_filter)
         continue;
      if(magic_filter >= 0 && HistoryDealGetInteger(ticket, DEAL_MAGIC) != magic_filter)
         continue;

      long     entry  = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      ulong    pos_id = (ulong)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      datetime t      = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);

      if(entry == DEAL_ENTRY_IN)
        {
         int k = ArraySize(g_entries);
         ArrayResize(g_entries, k + 1);
         g_entries[k].pos_id     = pos_id;
         g_entries[k].time       = t;
         g_entries[k].symbol     = sym;
         g_entries[k].commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         continue;
        }
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY)
        {
         n_skipped++;                                // reversals (INOUT) are not paired
         continue;
        }

      int e = FindEntry(pos_id);
      if(e < 0)
        {
         n_skipped++;                                // entry is older than the window
         continue;
        }

      double net = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                   + HistoryDealGetDouble(ticket, DEAL_SWAP)
                   + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
                   + g_entries[e].commission;
      g_entries[e].commission = 0.0;                 // count the entry commission once

      int bar_in  = iBarShift(sym, tf, g_entries[e].time, false);
      int bar_out = iBarShift(sym, tf, t, false);
      if(bar_in < 0 || bar_out < 0)
        {
         n_skipped++;                                // price history not loaded
         continue;
        }

      int bars = bar_in - bar_out;                   // 0 = closed inside the entry bar
      n_all++;
      sum_bars += bars;
      if(net > 0.0)
         gp_all += net;
      else
         gl_all -= net;

      if(bars == 0)
         n_same++;
      else
        {
         if(net > 0.0)
            gp_later += net;
         else
            gl_later -= net;
        }
     }

   if(n_all == 0)
     {
      Print("No closed positions found for the selected filters. Skipped deals: ", n_skipped);
      return false;
     }

   double same_share = 100.0 * n_same / n_all;
   double avg_bars   = sum_bars / n_all;

   PrintFormat("=== Same-bar exit check (%s, %s -> %s) ===",
               EnumToString(tf), TimeToString(from), TimeToString(to));
   PrintFormat("Closing deals analysed : %d   (skipped: %d)", n_all, n_skipped);
   PrintFormat("Closed in entry bar    : %d  (%.1f%%)", n_same, same_share);
   PrintFormat("Average bars in trade  : %.2f", avg_bars);
   PrintFormat("Profit factor, all            : %s", PfToString(ProfitFactor(gp_all, gl_all)));
   PrintFormat("Profit factor, after 1st bar  : %s", PfToString(ProfitFactor(gp_later, gl_later)));

   if(avg_bars < 1.0 || same_share > 50.0)
      Print("WARNING: most trades close inside the bar they were opened on. "
            "If these results came from a bar-based simulation, check that SL/TP/"
            "trailing are never evaluated against the entry bar's own High/Low.");
   return true;
  }
//+------------------------------------------------------------------+
