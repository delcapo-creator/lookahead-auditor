//+------------------------------------------------------------------+
//|                                             SameBarExitCheck.mq5 |
//|  Script: runs the same-bar exit check on this account's history. |
//+------------------------------------------------------------------+
#property copyright "Saad"
#property version   "1.00"
#property script_show_inputs

#include "SameBarExitCheck.mqh"

input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1; // Bar timeframe the strategy decides on
input int             InpDaysBack  = 365;       // History window, days
input string          InpSymbol    = "";        // Symbol filter (empty = all)
input long            InpMagic     = -1;        // Magic filter (-1 = all)

void OnStart()
  {
   datetime to   = TimeCurrent();
   datetime from = to - (datetime)((long)InpDaysBack * 86400);
   SameBarExitReport(InpTimeframe, from, to, InpSymbol, InpMagic);
  }
//+------------------------------------------------------------------+
