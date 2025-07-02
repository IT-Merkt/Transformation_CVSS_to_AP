# report.py
from datetime import datetime
import os

def save_html_report(metrics: dict, output_path: str, model_name="LoRA-Mistral"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Evaluation Report – {model_name}</title>
  <style>
    body {{
      font-family: sans-serif;
      padding: 2rem;
      background: #f4f4f4;
      color: #333;
    }}
    h1 {{
      color: #005b9f;
    }}
    table {{
      border-collapse: collapse;
      width: 50%;
      margin-top: 1rem;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 0.5rem;
      text-align: left;
    }}
    th {{
      background: #e0e0e0;
    }}
    .footer {{
      margin-top: 2rem;
      font-size: 0.9rem;
      color: #666;
    }}
  </style>
</head>
<body>
  <h1>Evaluation Report – {model_name}</h1>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Exact Match</td><td>{metrics['exact_match']:.2%}</td></tr>
    <tr><td>BLEU Score</td><td>{metrics['bleu']:.4f}</td></tr>
    <tr><td>ROUGE-L</td><td>{metrics['rougeL']:.4f}</td></tr>
    <tr><td>Levenshtein Distance</td><td>{metrics['levenshtein_distance']:.2f}</td></tr>
  </table>
  <div class="footer">
    Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 HTML-Report gespeichert: {output_path}")
