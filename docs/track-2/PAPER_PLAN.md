# Track 2 plan: the paper ("Burst to Flex", CCGrid 2027)

Version 1, 11 Nov 2026. Companion to `SIM_PLAN.md` (phases S0–S16). Figure and table IDs here are the ones used in SIM_PLAN.

**Fixed:**
- Abstract by 24 Nov 2026 AoE; full paper by 1 Dec 2026 AoE.
- 10 pages including references; IEEE conference template (`IEEEtran`, `conference` option); double-blind.

**Check on the submission site when it opens:**
- Can the abstract be edited between 24 Nov and 1 Dec? Many systems allow it, but don't assume.
- Is there a separate abstract word limit?

---

## 1. Sections and page budget (10.0 pages)

| § | Section | Pages | Content | Can write before results? |
|---|---|---|---|---|
| — | Title, abstract | 0.3 | ≤ 200 words | Skeleton yes, numbers no |
| 1 | Introduction | 1.0 | Grid stress at evening peaks; GPU clusters as flexible load; three levers; why bursting is under-studied as a grid action; why "which jobs" needs prediction; contributions (4 bullets); F1 teaser | Yes (all but F1 and the headline numbers) |
| 2 | Background and motivation | 1.0 | Demand response and $/MWh; Philly facts: many jobs fail or are killed, failed jobs die early, power has a daily shape → F2 | Text yes; F2 needs S10 |
| 3 | Problem formulation | 1.0 | Event (t_s, D, cut); target P*; power model (D5); actions and their costs (D7, D8, D9); $/MWh definition; assumptions box (D4, D8, D12) | **Yes, fully** |
| 4 | Design | 1.4 | Controller: greedy $/kW over cap levels plus job actions (D11); burst selection rule and why remaining runtime about the length of D and low failure risk are the sweet spot; predictor interface; transformer (small figure F4b, content from Track 1) | **Yes, fully** (transformer details from the Track 1 plan) |
| 5 | Methodology | 0.8 | Trace and test window; simulator (GPU pool, FCFS skip-ahead, paired runs, D6); parameters table T1 with sources; baselines (Tsafrir last-two, GBDT, LightGBM, oracle); events (D13) | Yes, except the final parameter values (S11) |
| 6 | Evaluation | 2.8 | 6.1 E1 cost of flexibility (F6, T2) · 6.2 E2 when burst wins (F7, T4) · 6.3 E3 does prediction help (F8, T3) · 6.4 E4 user impact (F9) · 6.5 E5 carbon (F10 or one paragraph) | No |
| 7 | Related work | 0.7 | Hybrid-cloud bursting (de Assunção HPDC'09, Bicer CCGrid'12, Clemente-Castelló CCGrid'17, Stratus SoCC'18, Starburst ATC'24, Eva EuroSys'25); data-center demand response (Liu SIGMETRICS'13, Zhou PESGM'24); carbon-aware computing (Radovanović TPS'23, Wiesner Middleware'21, CarbonScaler SIGMETRICS'24, Carbon Explorer ASPLOS'23); GPU power (Zeus NSDI'23, Patel ASPLOS'24); job-runtime prediction (Tsafrir TPDS'07, Hu SC'21, Lucid ASPLOS'23); GPU traces (Jeon ATC'19, Weng NSDI'22) | **Yes, fully** |
| 8 | Discussion, limitations, conclusion | 0.4 | Threats: simulation only, one trace, merged attempts, GPU model unknown, synthetic deadlines, checkpoint assumption, constant carbon; guidelines ("use cap for shallow short cuts, burst for …"); future work | Threats yes; guidelines no |
| — | References | 0.6 | About 25–30 entries | Yes |

If over length, cut in this order:
1. Shrink F1 or drop it (use F6 as the teaser).
2. E5 → one paragraph.
3. Merge F9 into T2.
4. Shorten §2.

## 2. Figures and tables

| ID | What | Source | Phase | Experiment |
|---|---|---|---|---|
| F1 | Teaser: facility power over one event (baseline, delay, cap, burst, mix) with target P* | e1 runs, one event | S12 | E1 |
| F2 | Motivation: (a) GPU-h and jobs by status; (b) CDF of runtime by status; (c) one week of baseline power with evening windows | trace + S8 baseline | S10 | — |
| F3 | System overview: trace → simulator → event → controller ← predictions; actions to cluster and cloud (drawn by hand: draw.io or TikZ) | — | W3 | — |
| F4 | (a) The controller's greedy loop; (b) the transformer (token = past job, [CUR] token, two heads). Can be one combined figure with F3 | Track 1 plan | W3 | — |
| F6 | **Main:** $/MWh vs event length, one panel per cut, lines for delay / cap / burst / mix (median, IQR band); shortfall marked | e1 summary | S12 | E1 |
| F7 | Winner map: cloud price multiplier (x) × boot delay (y), cell = cheapest policy, annotated with the mix's burst share | e2 summary | S13 | E2 |
| F8 | Burst selection: $/MWh and cloud $ wasted (stacked: Failed/Killed, short remaining, useful) per selector | e3 summary | S14 | E3 |
| F9 | User impact: Δwait and missed deadlines per policy (bars), and rebound energy | e4 summary | S12 | E4 |
| F10 | Carbon: local avoided vs cloud added kgCO2 per event for 3 cloud regions | e5 summary | S15 | E5 (trim) |
| T1 | Parameters with source and date (power, PUE, cap curve, price, boot, overheads, costs) | flex.yaml | S11 | — |
| T2 | Default event (2 h, 20%): shed MWh, compliance, $/MWh, cloud $, ΔGPU-h, misses, rebound per policy | e1 summary | S12 | E1/E4 |
| T3 | Predictor accuracy (pinball/MAE for remaining runtime, AUC for fail/cancel) next to the decision metric ($ wasted) | Track 1 reports + e3 | S14 | E3 |
| T4 | Sensitivity: $/MWh of each policy on the default event under each perturbation | e2 sensitivity | S13 | E2 |

Figure rules:
- One-column width (3.5 in), readable when printed in grey.
- Same colour and marker per policy in every figure.
- Every figure is made by a script in `paper/figures/`. No hand-edited numbers.

## 3. Writing order (W0–W8)

| # | Task | Needs | Effort (h) | Who |
|---|---|---|---|---|
| W0 | LaTeX skeleton: `paper/main.tex` (IEEEtran conference, anonymous author block), `sections/*.tex`, `refs.bib`, `numbers.tex` stub, a build command; `latexmk` or Overleaf. Compiles to about 1 page of placeholders | nothing | 1.5 | Claude Code (prompt in STARTER_PROMPTS_T2) |
| W1 | `refs.bib`: the 19 core references from PROJECT_CONTEXT §11, each checked on dblp (venue, year, pages) | nothing | 1.5 | review chat + you |
| W2 | §3 Formulation and §4 Design text | SIM_PLAN decisions | 3 | you + review chat |
| W3 | F3/F4 diagrams | §4 text | 1.5 | you |
| W4 | §7 Related work, §1 Intro (without numbers), §8 threats | W1 | 3 | you + review chat |
| W5 | Abstract v1 (see section 4); submit by 24 Nov | as much of S12 as exists | 1 | you + review chat |
| W6 | §2 Motivation, §5 Methodology | S10, S11 | 2 | you + review chat |
| W7 | §6 Evaluation text around the figures; guidelines in §8 | S12–S15 | 4 | you + review chat |
| W8 | Final pass: numbers via `numbers.tex` (S16), page fit, double-blind check, PDF/eXpress if the venue requires it | S16 | 2 | you + review chat |

W0–W4 need no results: about 10 h of writing that fits while Track 1 trains or experiments run.

## 4. Abstract plan (due 24 Nov)

Structure (about 180 words, 6 sentences):
1. **Problem:** grids ask large consumers to cut load at peaks; GPU clusters are large, fast-growing loads.
2. **Gap:** delay and power capping are studied; bursting jobs to rented cloud GPUs as a grid action, priced in $/MWh, is not. And which jobs to burst matters.
3. **Approach:** we price delay, power-cap and burst in $ per MWh shed on the Microsoft Philly trace, with a controller that mixes them and selects jobs using predicted remaining runtime and failure risk from a small transformer over each user's job history.
4. **Result 1:** [E1 number: the mix's $/MWh vs the best single action at the default event].
5. **Result 2:** [E2 condition: when burst wins] and [E3: the predictor's cloud-dollar saving vs no predictor / last-two].
6. **Takeaway:** guidelines for operators.

Rules:
- **Only numbers that exist on 24 Nov.** If S12 has run with last2/oracle, use E1 numbers and name the predictor honestly ("with a history-based predictor"). If the transformer results are not in, sentence 5 says "we evaluate…" instead of "we show the transformer reduces…".
- If no Philly results exist by 23 Nov: a qualitative abstract (sentences 1–3 and 6, plus "we quantify…"), with no invented numbers and no synthetic numbers.
- Keep the claims the 1 Dec paper can still change small (direction words, not exact percentages), unless the abstract can't be edited later.

## 5. References plan
- Use the core list in PROJECT_CONTEXT §11.
- Still to verify on dblp before citing: Mu'alem and Feitelson (TPDS 2001), Feitelson and Rudolph (JSSPP 1998), Perseus (SOSP 2024), Li et al. (SC 2023), Vaswani et al. (NeurIPS 2017), Ke et al. LightGBM (NeurIPS 2017).
- Candidates for S11 parameters (verify the venue first):
  - A published VM start-up time study (for example Mao and Humphrey, IEEE CLOUD 2012).
  - A published DNN checkpointing study (for example CheckFreq, USENIX FAST 2021).
- **Open question for your faculty:** datasheets (GPU power), cloud price pages and carbon-intensity data are not papers. The usual practice is a footnote with the URL and access date. Confirm this is acceptable under the "published references only" rule.

## 6. Double-blind checklist (check at W8)
- No author names, university, city, acknowledgements or funding in the PDF. Check the PDF metadata too (`\hypersetup{pdfauthor={}}`).
- No GitHub link. Artifact: "available upon acceptance", or an anonymised mirror (Later).
- No "our previous work [x]" self-references.
- Figures carry no usernames or file paths.

## 7. Later (after the first full draft)
- An anonymised artifact link.
- A related-work comparison table (bursting / DR / carbon / prediction columns).
- Extra plots: per-event scatter of $/MWh vs event start hour.
- Comparison with the published demand-response compensation level ($/MWh).
- Poster version (2–4 pages, 15 Feb) as the fallback if 1 Dec slips.
