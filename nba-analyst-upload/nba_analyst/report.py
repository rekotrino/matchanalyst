"""Markdown pre-match reports: one file per game plus a daily digest."""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import SETTINGS
from .ingest.espn import TEAM_NAMES
from .model.game_model import fair_markets

ET = ZoneInfo("America/New_York")


def _pct(p):
    return "-" if p is None else f"{100 * p:.1f}%"


def _f(v, nd=1, signed=False):
    if v is None:
        return "-"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _tip(iso_utc):
    dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ET).strftime("%a %b %d, %I:%M %p ET")


def game_report(res):
    g, pred, lg = res["game"], res["pred"], None
    home, away = g["home"], g["away"]
    L = []
    L.append(f"# {TEAM_NAMES.get(away, away)} @ {TEAM_NAMES.get(home, home)}")
    L.append(f"*{_tip(g['date_utc'])} · game {g['game_id']} · profiles from {res['profile_source']}*\n")

    # --- model summary
    L.append("## Model view")
    mo = pred.get("model_only", pred)
    w = pred.get("anchor_weight", 1.0)
    L.append(f"- Pure model: {home} {mo['exp_home_pts']:.1f} – {away} {mo['exp_away_pts']:.1f} "
             f"(margin {home} {_f(mo['exp_margin'], 1, True)}, total {mo['exp_total']:.1f}, pace {pred['pace']:.1f}, "
             f"{home} win {_pct(mo['p_home_win'])})")
    if res["ref"]:
        r = res["ref"]
        rp = res["ref_probs"]
        L.append(f"- Reference line ({r['provider']}): {home} {_f(r['spread_home'], 1, True)}, total {_f(r['total'])}"
                 + (f", implied {home} {_pct(rp['p_home'])} after removing vig" if rp else ""))
        if rp and abs(rp["p_home"] - mo["p_home_win"]) > 0.10:
            L.append(f"- ⚠ Pure model disagrees with the reference line by {100 * abs(rp['p_home'] - mo['p_home_win']):.0f} pts. "
                     "Usually a roster change or injury the ratings haven't seen yet.")
    L.append(f"- **Working numbers** (model weight {w:.0%}, reference {1 - w:.0%}): "
             f"**{home} {pred['exp_home_pts']:.1f} – {away} {pred['exp_away_pts']:.1f}**, "
             f"margin {home} {_f(pred['exp_margin'], 1, True)}, total {pred['exp_total']:.1f}, "
             f"**{home} win {_pct(pred['p_home_win'])}** / {away} {_pct(1 - pred['p_home_win'])}")
    for n in pred["notes"]:
        L.append(f"- Schedule: {n}")
    L.append("")

    # --- teams
    for t in (home, away):
        prof, iss, rt, rest = res["profiles"][t], res["issues"][t], res["ratings"].get(t) or {}, res["rest"][t]
        L.append(f"## {TEAM_NAMES.get(t, t)} ({'home' if t == home else 'away'})")
        l5, l10, s = prof["last5"], prof["last10"], prof["season"]
        L.append(f"- Strength rating: offense {_f(rt.get('off'), 1, True)}, defense {_f(rt.get('def'), 1, True)} "
                 f"(pts/100 vs league; defense negative = better), pace {_f(rt.get('pace'), 1, True)}; "
                 f"based on {rt.get('n', 0)} games this season" + (" + last-season prior" if res['profile_source'].startswith('current') else ""))
        if l10["n"]:
            L.append(f"- Last 10: **{l10['record']}**, margin {_f(l10['margin'], 1, True)}, ORtg {_f(l10['ortg'])}, DRtg {_f(l10['drtg'])}, "
                     f"eFG {_pct(l10['efg'])} / opp eFG {_pct(l10['opp_efg'])}, TOV {_pct(l10['tov_pct'])}, "
                     f"OREB {_pct(l10['oreb_pct'])}, 3P rate {_pct(l10['tp_rate'])}")
            L.append(f"- Last 5: **{l5['record']}**, margin {_f(l5['margin'], 1, True)}, ORtg {_f(l5['ortg'])}, DRtg {_f(l5['drtg'])}")
            if prof["home_net"] is not None or prof["away_net"] is not None:
                L.append(f"- Home margin {_f(prof['home_net'], 1, True)} · Away margin {_f(prof['away_net'], 1, True)}")
        if rest.get("rest_days") is not None:
            tag = " **(back-to-back)**" if rest["b2b"] else (" (3rd in 4 nights)" if rest["three_in_four"] else "")
            L.append(f"- Rest: {rest['rest_days']} day(s){tag}, {rest['games_last7']} games in last 7 days")
        strengths = [x for x in iss if x[0] == "strength"]
        issues = [x for x in iss if x[0] == "issue"]
        notes = [x for x in iss if x[0] == "note"]
        if strengths:
            L.append("- Strengths: " + "; ".join(x[2] for x in strengths))
        if issues:
            L.append("- Issues: " + "; ".join(x[2] for x in issues))
        for x in notes:
            L.append(f"- Note: {x[2]}")
        inj = [i for i in res["injuries"] if i["team"] == t]
        if inj:
            L.append("- Injury report: " + "; ".join(f"{i['player']} ({i['status']})" for i in inj))
        if prof["last_games"]:
            L.append("- Recent: " + ", ".join(
                f"{'vs' if x['home'] else '@'} {x['opp']} {'W' if x['margin'] > 0 else 'L'} {x['pts']}-{x['opp_pts']}"
                for x in prof["last_games"][-5:]))
        L.append("")

    # --- player rotations
    L.append("## Player analytics (last 10 games with this team)")
    for t in (home, away):
        rot = res.get("rotations", {}).get(t, [])
        L.append(f"### {TEAM_NAMES.get(t, t)}")
        if not rot:
            L.append("_No player box scores stored for this team yet._\n")
            continue
        outs = {n.lower() for n in res.get("out_names", {}).get(t, [])}
        L.append("| Player | Pos | G | MIN | PTS | REB | AST | PRA | L5 PTS/REB/AST | Trend (L5−L10) | Note |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in rot:
            l10, l5, tr = p["last10"], p["last5"], p["trend"]
            note = []
            if p["player"].lower() in outs:
                note.append("**OUT**")
            if p["starter_rate"] >= 0.8:
                note.append("starter")
            if tr["min"] >= 4:
                note.append("minutes up")
            elif tr["min"] <= -4:
                note.append("minutes down")
            if l10.get("pts_sd") and l10["pts"] >= 10 and l10["pts_sd"] / l10["pts"] > 0.5:
                note.append("volatile scorer")
            L.append(f"| {p['player']} | {p['position'] or ''} | {p['games']} | {l10['min']:.1f} | {l10['pts']:.1f} | "
                     f"{l10['reb']:.1f} | {l10['ast']:.1f} | {l10['pra']:.1f} | "
                     f"{l5['pts']:.1f}/{l5['reb']:.1f}/{l5['ast']:.1f} | "
                     f"{tr['pts']:+.1f}/{tr['reb']:+.1f}/{tr['ast']:+.1f} | {', '.join(note)} |")
        L.append("")

    # --- fair price grid
    L.append("## Fair prices from the model")
    L.append("| Market | Our probability |")
    L.append("|---|---|")
    for row in fair_markets(pred, home, away):
        L.append(f"| {row['label']} | {_pct(row['p'])} |")
    L.append("")

    # --- market scan
    L.append("## Polymarket scan")
    modelled = [m for m in res["markets"] if m["rows"]]
    other = [m for m in res["markets"] if not m["rows"]]
    if not res["markets"]:
        L.append("_No Polymarket event found for this game yet._")
    if modelled:
        L.append("| Market | Outcome | Our prob | Buy price | Edge | Liquidity | Verdict |")
        L.append("|---|---|---|---|---|---|---|")
        cfg = SETTINGS.betting
        for m in modelled:
            mk = m["market"]
            for r in m["rows"]:
                if r["edge"] >= cfg.edge_threshold and r["our_prob"] >= cfg.prob_floor and (mk.get("liquidity") or 0) >= cfg.min_liquidity:
                    verdict = "**candidate**"
                elif r["edge"] >= cfg.edge_threshold and r["our_prob"] >= cfg.prob_floor:
                    verdict = "edge but illiquid"
                elif r["edge"] >= cfg.edge_threshold:
                    verdict = "edge but below prob floor"
                elif r["edge"] > 0:
                    verdict = "thin edge"
                else:
                    verdict = "no edge"
                out_label = r["outcome"]
                if r.get("proj") is not None:
                    hr = f", L10 hit {100 * r['hit_rate']:.0f}%" if r.get("hit_rate") is not None else ""
                    out_label = f"{r['outcome']} (proj {r['proj']:.1f} ± {r['proj_sd']:.1f}, {r['proj_min']:.0f} min{hr})"
                L.append(f"| {mk.get('question') or mk['market_type']} | {out_label} | {_pct(r['our_prob'])} | "
                         f"{r['buy_price']:.3f} | {100 * r['edge']:+.1f} | ${(mk.get('liquidity') or 0):,.0f} | {verdict} |")
    if other:
        kinds = {}
        for m in other:
            kinds[m["market"]["market_type"]] = kinds.get(m["market"]["market_type"], 0) + 1
        L.append(f"\n_Listed but not priced (novelty markets, or props for players without enough games with this team): "
                 + ", ".join(f"{k} ×{v}" for k, v in kinds.items()) + "._")
    L.append("")

    # --- picks
    L.append("## Picks (paper)")
    if res["picks"]:
        for p in res["picks"]:
            L.append(f"- **{p['question']} → {p['outcome']}** at {p['buy_price']:.3f} · our prob {_pct(p['our_prob'])} · "
                     f"edge {100 * p['edge']:+.1f} pts · stake ${p['stake']:.2f}")
    else:
        L.append("_No bet. Nothing on this game clears the edge threshold with enough liquidity._")
    L.append("")
    return "\n".join(L)


def daily_digest(ctx, results, perf):
    today = datetime.now(ET).strftime("%Y-%m-%d")
    L = [f"# NBA Analyst digest — {today}", ""]
    L.append(f"Season {ctx['season'] - 1}-{str(ctx['season'])[2:]}: {ctx['n_cur_games']} games this season in the store, "
             f"{ctx['n_prior_games']} prior-season games as the baseline. "
             f"Betting config: edge ≥ {100 * SETTINGS.betting.edge_threshold:.1f} pts, prob ≥ {100 * SETTINGS.betting.prob_floor:.0f}%, "
             f"liquidity ≥ ${SETTINGS.betting.min_liquidity:,.0f}, quarter-Kelly, bankroll ${SETTINGS.betting.bankroll:,.0f}.")
    L.append("")
    L.append("## Today's board")
    L.append("| Game | Tip (ET) | Working score | Home win | Pure model home win | Ref line | Picks |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        g, p = r["game"], r["pred"]
        ref = r["ref"]
        mo = p.get("model_only", p)
        refs = f"{g['home']} {_f(ref['spread_home'], 1, True)} / {_f(ref['total'])}" if ref else "-"
        L.append(f"| {g['away']} @ {g['home']} | {_tip(g['date_utc'])} | {p['exp_home_pts']:.0f}-{p['exp_away_pts']:.0f} | "
                 f"{_pct(p['p_home_win'])} | {_pct(mo['p_home_win'])} | {refs} | {len(r['picks'])} |")
    L.append("")
    picks = [p for r in results for p in r["picks"]]
    L.append("## Picks (paper)")
    if picks:
        L.append("| Game | Market | Outcome | Our prob | Price | Edge | Stake |")
        L.append("|---|---|---|---|---|---|---|")
        for p in picks:
            L.append(f"| {p['matchup']} | {p['question']} | {p['outcome']} | {_pct(p['our_prob'])} | "
                     f"{p['buy_price']:.3f} | {100 * p['edge']:+.1f} | ${p['stake']:.2f} |")
    else:
        L.append("_No picks today._")
    L.append("")
    L.append("## Paper record so far")
    if perf.get("n"):
        L.append(f"{perf['n']} graded picks, {perf['wins']} wins ({100 * perf['hit_rate']:.0f}%), "
                 f"P&L ${perf['pnl']:+.2f} (ROI {100 * perf['roi']:+.1f}%), "
                 f"avg CLV {('%+.1f pts' % (100 * perf['avg_clv'])) if perf['avg_clv'] is not None else 'n/a'} over {perf['n_clv']} picks.")
    else:
        L.append("_No graded picks yet._")
    L.append("")
    L.append("_Phase 0 model: uncalibrated. Picks are paper only. Reports are for sanity-checking the engine, not for betting._")
    return "\n".join(L)


def write_reports(ctx, results, perf, out_dir=None):
    out_dir = out_dir or SETTINGS.reports_dir
    today = datetime.now(ET).strftime("%Y-%m-%d")
    day_dir = os.path.join(out_dir, today)
    os.makedirs(day_dir, exist_ok=True)
    paths = []
    for r in results:
        g = r["game"]
        p = os.path.join(day_dir, f"{g['date_et']}_{g['away']}_at_{g['home']}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(game_report(r))
        paths.append(p)
    digest = os.path.join(out_dir, f"{today}_digest.md")
    with open(digest, "w", encoding="utf-8") as f:
        f.write(daily_digest(ctx, results, perf))
    return digest, paths
