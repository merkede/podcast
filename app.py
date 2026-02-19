from flask import Flask, render_template_string


app = Flask(__name__)


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Transfer Dashboard Home</title>
  <style>
    :root {
      --bg: #f3f2f1;
      --card: #ffffff;
      --text: #201f1e;
      --muted: #605e5c;
      --primary: #00bcf2;
      --secondary: #742774;
      --shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
      background: linear-gradient(160deg, #eef7fb 0%, var(--bg) 60%, #f8f3fb 100%);
      color: var(--text);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    .header {
      background: var(--card);
      border-radius: 12px;
      box-shadow: var(--shadow);
      padding: 18px 20px;
      margin-bottom: 20px;
      border-left: 5px solid var(--primary);
    }
    .title {
      margin: 0;
      font-size: 1.8rem;
      color: #0078d4;
      font-weight: 700;
    }
    .sub {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--card);
      border-radius: 12px;
      box-shadow: var(--shadow);
      padding: 18px;
      border: 1px solid #ecebe9;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .card h2 {
      margin: 0;
      font-size: 1.15rem;
    }
    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 0.92rem;
    }
    .meta {
      font-size: 0.82rem;
      color: #8a8886;
      margin-top: auto;
    }
    .btn {
      display: inline-block;
      text-align: center;
      text-decoration: none;
      color: #fff;
      background: linear-gradient(135deg, #00bcf2, #0078d4);
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 600;
      font-size: 0.9rem;
    }
    .btn.secondary {
      background: linear-gradient(135deg, #8764b8, var(--secondary));
    }
    .note {
      margin-top: 20px;
      background: #ffffff;
      border-radius: 10px;
      padding: 14px 16px;
      border-left: 4px solid #f2c80f;
      color: #444;
      box-shadow: var(--shadow);
      font-size: 0.9rem;
      line-height: 1.45;
    }
    code {
      background: #f3f2f1;
      border-radius: 4px;
      padding: 2px 6px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="header">
      <h1 class="title">Transfer Dashboard Home</h1>
      <p class="sub">Select an analytics experience based on audience and objective.</p>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Messenger Transfer Analytics</h2>
        <p>Comprehensive operational view for Messenger case routing: transfer behavior, process flow, effort impact, and journey diagnostics. Best for broad performance monitoring and stakeholder reporting.</p>
        <div class="meta">Expected URL: <code>http://127.0.0.1:8050</code></div>
        <a class="btn" href="http://127.0.0.1:8050" target="_blank" rel="noopener noreferrer">Open Original Dashboard</a>
      </article>

      <article class="card">
        <h2>Telephony Transfer Analytics</h2>
        <p>Telephony-focused deep-dive into transfer pathways, uplift, and leakage drivers. Designed to surface high-cost journey patterns, quantify transfer penalties, and support targeted improvement actions.</p>
        <div class="meta">Expected URL: <code>http://127.0.0.1:8051</code></div>
        <a class="btn secondary" href="http://127.0.0.1:8051" target="_blank" rel="noopener noreferrer">Open Telephony Dashboard</a>
      </article>
    </section>

    <section class="note">
      If a link does not load, start that dashboard first, then refresh this page.<br/>
      Landing page run command: <code>python3 app.py</code>
    </section>
  </div>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(HTML)


if __name__ == "__main__":
    app.run(debug=True, port=8060)
