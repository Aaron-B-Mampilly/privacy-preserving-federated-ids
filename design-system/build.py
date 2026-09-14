"""Generates the Claude Design preview bundle for the FedPDA-IDS UI.

Each output file is a standalone HTML card that mirrors ONE group of the
React component library in frontend/src/components. The tokens below are
the same values Tailwind consumes from frontend/src/styles/tokens.css --
they are duplicated here only because a Design preview has to be
self-contained HTML, and this script exists so the duplication is
generated from one place rather than hand-maintained in eight files.

Run:  python design-system/build.py
Then: push with the DesignSync tool (requires /design-login once).
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent

TOKENS = """
:root {
  --bg:#f5f6f8; --surface:#fff; --surface-alt:#eef1f4; --ink:#161b26;
  --ink-muted:#5c6478; --border:#dde1e8; --accent:#0d6d67; --accent-strong:#0a4f4b;
  --accent-soft:#e2f2f0; --finding:#a34e0e; --finding-soft:#fbecdf; --caveat:#8a3b52;
  --caveat-soft:#f8e9ee; --good:#1a7a4c; --good-soft:#e4f3ea; --danger:#b02a2a;
  --danger-soft:#fbe8e8;
  --shadow:0 1px 2px rgba(22,27,38,.04), 0 4px 16px rgba(22,27,38,.05);
}
* { box-sizing:border-box; }
body {
  margin:0; padding:28px; background:var(--bg); color:var(--ink);
  font-family:'IBM Plex Sans',system-ui,sans-serif; font-size:15px; line-height:1.6;
}
h1,h2,h3,h4 { font-family:Fraunces,Georgia,serif; font-weight:600; margin:0; }
.mono,code,td,th,.num { font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px;
        box-shadow:var(--shadow); padding:20px; }
.label-caps { font-size:11px; font-weight:600; text-transform:uppercase;
              letter-spacing:.06em; color:var(--ink-muted); }
.row { display:flex; flex-wrap:wrap; gap:12px; align-items:flex-start; }
.grid { display:grid; gap:12px; }
.spec { margin-top:10px; font-size:12px; color:var(--ink-muted); }
section { margin-bottom:26px; }
"""

FONTS = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Sans:wght@400;500;600'
    '&family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
)


def page(card_group: str, title: str, body: str) -> str:
    return f"""<!-- @dsCard group="{card_group}" -->
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>{FONTS}
<style>{TOKENS}</style></head>
<body>{body}</body></html>
"""


def write(name: str, group: str, title: str, body: str) -> None:
    (OUT / name).write_text(page(group, title, body), encoding="utf-8")
    print(f"wrote {name}")


# --------------------------------------------------------------- foundations

COLORS = [
    ("accent", "#0d6d67", "Primary actions, active nav, chart series 1"),
    ("accent-strong", "#0a4f4b", "Metric values, hover state"),
    ("ink", "#161b26", "Body text"),
    ("ink-muted", "#5c6478", "Secondary text, axis labels"),
    ("border", "#dde1e8", "Card and table rules"),
    ("finding", "#a34e0e", "Findings, derived provenance"),
    ("caveat", "#8a3b52", "Caveats, methodology warnings"),
    ("good", "#1a7a4c", "Healthy status, live provenance"),
    ("danger", "#b02a2a", "Errors, disconnected state"),
]

swatches = "".join(
    f'<div class="card" style="padding:0;overflow:hidden">'
    f'<div style="height:56px;background:{hexv}"></div>'
    f'<div style="padding:10px 12px"><div class="mono" style="font-size:12px">{name}</div>'
    f'<div class="mono" style="font-size:11px;color:var(--ink-muted)">{hexv}</div>'
    f'<div class="spec">{use}</div></div></div>'
    for name, hexv, use in COLORS
)

write("01-foundations.html", "Foundations", "Colour & type", f"""
<h2>Foundations</h2>
<p class="spec">Palette carried over from the project's published results artifact so the dashboard
and the thesis write-up read as one system.</p>
<section>
  <div class="label-caps" style="margin-bottom:10px">Colour</div>
  <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(180px,1fr))">{swatches}</div>
</section>
<section>
  <div class="label-caps" style="margin-bottom:10px">Type</div>
  <div class="card">
    <h1 style="font-size:32px">Fraunces · display</h1>
    <p class="spec">Headings and metric values. Weight 600.</p>
    <p style="font-size:15px;margin:14px 0 0">IBM Plex Sans · body copy at 15px/1.6 for readable prose.</p>
    <p class="mono" style="font-size:14px;margin-top:10px">IBM Plex Mono · 0.0831 ± 0.0001 — tabular numerics</p>
    <p class="spec">Digits use tabular-nums everywhere so table columns align.</p>
  </div>
</section>""")

# ------------------------------------------------------------------- buttons

write("02-buttons.html", "Actions", "Buttons", """
<h2>Buttons</h2>
<section><div class="row">
  <button style="background:var(--accent);color:#fff;border:1px solid transparent;border-radius:6px;
    padding:6px 12px;font:500 14px 'IBM Plex Sans';cursor:pointer">Primary</button>
  <button style="background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:6px;
    padding:6px 12px;font:500 14px 'IBM Plex Sans';cursor:pointer">Secondary</button>
  <button style="background:transparent;color:var(--ink);border:1px solid transparent;border-radius:6px;
    padding:6px 12px;font:500 14px 'IBM Plex Sans';cursor:pointer">Ghost</button>
  <button disabled style="background:var(--accent);opacity:.5;color:#fff;border:1px solid transparent;
    border-radius:6px;padding:6px 12px;font:500 14px 'IBM Plex Sans'">Disabled</button>
</div>
<p class="spec">6px radius, 1px border on every variant so they share one silhouette.
Primary is reserved for the single main action on a screen (e.g. Analyze Traffic, Trigger retrain).</p>
</section>""")

# --------------------------------------------------------- cards & stat tiles

write("03-cards.html", "Cards", "Cards & stat tiles", """
<h2>Cards &amp; stat tiles</h2>
<section>
  <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(230px,1fr))">
    <div class="card" style="padding:16px">
      <div class="label-caps">Macro-F1</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-top:8px">
        <span class="num" style="font-family:Fraunces;font-size:26px;font-weight:600;color:var(--accent-strong)">0.0831</span>
        <span class="mono" style="font-size:12px;color:var(--ink-muted)">± 0.0001</span>
      </div>
      <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
        <span style="background:var(--surface-alt);color:var(--ink-muted);border-radius:4px;
          padding:1px 7px;font-size:11px;font-weight:600;text-transform:uppercase">stored</span>
        <span style="font-size:11px;color:var(--ink-muted)">n=40</span>
      </div>
    </div>
    <div class="card" style="padding:16px">
      <div class="label-caps">Rare-class recall</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-top:8px">
        <span class="num" style="font-family:Fraunces;font-size:26px;font-weight:600;color:var(--accent-strong)">0.0000</span>
      </div>
      <div style="margin-top:8px">
        <span style="background:var(--finding-soft);color:var(--finding);border-radius:4px;
          padding:1px 7px;font-size:11px;font-weight:600;text-transform:uppercase">derived</span>
      </div>
    </div>
    <div class="card" style="padding:16px">
      <div class="label-caps">Zero-day detection</div>
      <div style="margin-top:8px"><span style="font-family:Fraunces;font-size:26px;font-weight:600;color:var(--ink-muted)">—</span></div>
      <p class="spec" style="margin-top:6px">Not measured for this scope.</p>
    </div>
  </div>
  <p class="spec">A null metric renders as an em-dash with an explicit reason. It is never shown as 0 —
  fabricating a measured zero would misreport the experiment.</p>
</section>""")

# ----------------------------------------------------------- badges/provenance

write("04-badges.html", "Status", "Badges & provenance tags", """
<h2>Badges &amp; provenance</h2>
<section>
  <div class="label-caps" style="margin-bottom:8px">Provenance</div>
  <div class="row">
    <span style="background:var(--surface-alt);color:var(--ink-muted);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;text-transform:uppercase">stored</span>
    <span style="background:var(--accent-soft);color:var(--accent-strong);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;text-transform:uppercase">new run</span>
    <span style="background:var(--finding-soft);color:var(--finding);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;text-transform:uppercase">derived</span>
    <span style="background:var(--good-soft);color:var(--good);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;text-transform:uppercase">live</span>
    <span style="background:var(--caveat-soft);color:var(--caveat);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;text-transform:uppercase">demo data</span>
  </div>
  <p class="spec">Every metric in the product carries one of these. It is a research-integrity device:
  a reader can always tell a stored experiment result from a live computation from a demo fallback.</p>
</section>
<section>
  <div class="label-caps" style="margin-bottom:8px">Status badges</div>
  <div class="row">
    <span style="border:1px solid var(--border);background:var(--surface-alt);color:var(--ink-muted);border-radius:999px;padding:2px 10px;font-size:12px">neutral</span>
    <span style="border:1px solid rgba(13,109,103,.3);background:var(--accent-soft);color:var(--accent-strong);border-radius:999px;padding:2px 10px;font-size:12px">accent</span>
    <span style="border:1px solid rgba(26,122,76,.3);background:var(--good-soft);color:var(--good);border-radius:999px;padding:2px 10px;font-size:12px">healthy</span>
    <span style="border:1px solid rgba(163,78,14,.3);background:var(--finding-soft);color:var(--finding);border-radius:999px;padding:2px 10px;font-size:12px">warning</span>
    <span style="border:1px solid rgba(176,42,42,.3);background:var(--danger-soft);color:var(--danger);border-radius:999px;padding:2px 10px;font-size:12px">critical</span>
  </div>
</section>
<section>
  <div class="label-caps" style="margin-bottom:8px">Connection indicator</div>
  <div class="row">
    <span style="border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:4px 10px;font-size:12px;font-weight:500">
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--good)"></span>
      <span style="color:var(--good)"> Connected</span></span>
    <span style="border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:4px 10px;font-size:12px;font-weight:500">
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--finding)"></span>
      <span style="color:var(--finding)"> Connecting</span></span>
    <span style="border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:4px 10px;font-size:12px;font-weight:500">
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--danger)"></span>
      <span style="color:var(--danger)"> Disconnected</span></span>
  </div>
</section>""")

# ------------------------------------------------------------------ callouts

write("05-callouts.html", "Callouts", "Callouts", """
<h2>Callouts</h2>
<p class="spec">The same four-way vocabulary the project's results artifact uses, so a claim carries
the same weight in the dashboard as it does in the write-up.</p>
<section class="grid" style="gap:14px">
  <div style="background:var(--finding-soft);border:1px solid rgba(163,78,14,.25);border-radius:10px;padding:16px">
    <div class="label-caps" style="color:var(--finding);margin-bottom:6px">Finding</div>
    <div style="font-size:14px">Rare-class recall drops to exactly 0.000 at every finite ε — a measured
    zero, not a missing value.</div></div>
  <div style="background:var(--caveat-soft);border:1px solid rgba(138,59,82,.25);border-radius:10px;padding:16px">
    <div class="label-caps" style="color:var(--caveat);margin-bottom:6px">Caveat</div>
    <div style="font-size:14px">Only 8 of 45 clients had pre-cutoff data; all three arms are restricted
    to that matched subset so the comparison is not confounded.</div></div>
  <div style="background:var(--surface-alt);border:1px solid var(--border);border-radius:10px;padding:16px">
    <div class="label-caps" style="margin-bottom:6px">Mechanism</div>
    <div style="font-size:14px">Training runs offline in multi-hour batches, so round-by-round charts
    replay a completed run rather than streaming a live trainer.</div></div>
  <div style="background:var(--good-soft);border:1px solid rgba(26,122,76,.25);border-radius:10px;padding:16px">
    <div class="label-caps" style="color:var(--good);margin-bottom:6px">Resolved</div>
    <div style="font-size:14px">SecAgg+ reproduces the non-DP baseline to within quantization noise —
    the mechanism behaves as designed.</div></div>
</section>""")

# -------------------------------------------------------------------- tables

write("06-tables.html", "Data display", "Tables", """
<h2>Tables</h2>
<section>
<div style="border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--surface)">
<table style="width:100%;border-collapse:collapse;font-size:13.5px">
<thead><tr>
  <th style="text-align:left;padding:9px 14px;background:var(--surface-alt);border-bottom:1px solid var(--border);
    font:600 11px 'IBM Plex Sans';text-transform:uppercase;letter-spacing:.04em;color:var(--ink-muted)">Comparator</th>
  <th style="text-align:right;padding:9px 14px;background:var(--surface-alt);border-bottom:1px solid var(--border);
    font:600 11px 'IBM Plex Sans';text-transform:uppercase;letter-spacing:.04em;color:var(--ink-muted)">Macro-F1</th>
  <th style="text-align:right;padding:9px 14px;background:var(--surface-alt);border-bottom:1px solid var(--border);
    font:600 11px 'IBM Plex Sans';text-transform:uppercase;letter-spacing:.04em;color:var(--ink-muted)">Rare recall</th>
  <th style="text-align:right;padding:9px 14px;background:var(--surface-alt);border-bottom:1px solid var(--border);
    font:600 11px 'IBM Plex Sans';text-transform:uppercase;letter-spacing:.04em;color:var(--ink-muted)">MB/round</th>
</tr></thead>
<tbody>
  <tr><td style="padding:9px 14px;border-bottom:1px solid var(--border);font-weight:500">FedAvg</td>
      <td class="num" style="padding:9px 14px;border-bottom:1px solid var(--border);text-align:right">0.0831</td>
      <td class="num" style="padding:9px 14px;border-bottom:1px solid var(--border);text-align:right">0.0000</td>
      <td class="num" style="padding:9px 14px;border-bottom:1px solid var(--border);text-align:right">6.83</td></tr>
  <tr><td style="padding:9px 14px;border-bottom:1px solid var(--border);font-weight:500">Ours (no DP)</td>
      <td class="num" style="padding:9px 14px;border-bottom:1px solid var(--border);text-align:right">0.0831 <span style="color:var(--ink-muted)">± 0.0001</span></td>
      <td class="num" style="padding:9px 14px;border-bottom:1px solid var(--border);text-align:right">0.0000</td>
      <td class="num" style="padding:9px 14px;border-bottom:1px solid var(--border);text-align:right">6.75</td></tr>
  <tr><td style="padding:9px 14px;font-weight:500">Ours + DP</td>
      <td class="num" style="padding:9px 14px;text-align:right">0.0818 <span style="color:var(--ink-muted)">± 0.0021</span></td>
      <td style="padding:9px 14px;text-align:right;font-size:11px;text-transform:uppercase;color:var(--ink-muted)">not measured</td>
      <td class="num" style="padding:9px 14px;text-align:right">6.75</td></tr>
</tbody></table></div>
<p class="spec">Numeric columns are right-aligned and monospaced so digits line up.
A cell that was never measured says so in words rather than rendering a zero.</p>
</section>""")

# ------------------------------------------------------------ form controls

write("07-forms.html", "Forms", "Form controls", """
<h2>Form controls</h2>
<section><div class="row" style="align-items:center">
  <select style="border:1px solid var(--border);background:var(--surface);border-radius:6px;padding:5px 9px;
    font:500 13px 'IBM Plex Sans';color:var(--ink)"><option>CICIDS2017</option><option>N-BaIoT</option></select>
  <select class="mono" style="border:1px solid var(--border);background:var(--surface);border-radius:6px;
    padding:5px 9px;font-size:13px;color:var(--ink)"><option>α = 5</option><option>α = 0.5</option></select>
  <input placeholder="Filter by class" style="border:1px solid var(--border);background:var(--surface);
    border-radius:6px;padding:6px 10px;font:400 14px 'IBM Plex Sans';min-width:200px">
</div>
<p class="spec">The dataset selector is the only place the active federation is chosen. CICIDS2017 and
N-BaIoT are never merged, so there is deliberately no "all datasets" option.</p>
</section>
<section>
  <div class="label-caps" style="margin-bottom:8px">File upload</div>
  <div style="border:1.5px dashed var(--border);border-radius:10px;padding:26px;text-align:center;background:var(--surface)">
    <div style="font-size:14px;font-weight:500">Drop a preprocessed CSV window</div>
    <div class="spec" style="margin-top:4px">10 rows × 70 features for CICIDS2017 · validated before inference</div>
  </div>
</section>""")

# ------------------------------------------------------- states & indicators

write("08-states.html", "States", "Loading, empty & error states", """
<h2>States</h2>
<section class="grid" style="gap:14px">
  <div class="card" style="text-align:center;color:var(--ink-muted);font-size:14px;padding:34px">Loading…</div>
  <div class="card" style="text-align:center;padding:34px">
    <div style="font-size:14px;font-weight:500">No rows to display</div>
    <div class="spec">This run did not record these values.</div></div>
  <div style="background:var(--danger-soft);border:1px solid rgba(176,42,42,.3);border-radius:10px;padding:18px">
    <div style="font-weight:600;color:var(--danger);font-size:14px">Backend unavailable</div>
    <div style="font-size:14px;margin-top:4px">Could not reach the FedPDA-IDS API.</div>
    <div class="spec" style="margin-top:6px">Start the FastAPI server (uvicorn api.main:app --port 8001) and try again.</div>
  </div>
</section>
<p class="spec">Errors state what happened, then how to fix it. A Python traceback is never shown to the
user — the API refuses to send one and the client refuses to render one.</p>""")

print("\nBundle ready in design-system/. Push with the DesignSync tool after /design-login.")
