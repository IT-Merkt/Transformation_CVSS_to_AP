import pandas as pd
from datetime import datetime
from jinja2 import Template
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter

CSV_INPUT = "../outputs/summary_log.csv"
HTML_OUTPUT = "../outputs/results_report.html"
PLOT_DIR = "../outputs/"

# Load data
df = pd.read_csv(CSV_INPUT)

# Standardize column names
rename_dict = {}
if "GPT4o-score" in df.columns:
    rename_dict["GPT4o-score"] = "gpt_score"
if "DeppseekV3-score" in df.columns:
    rename_dict["DeppseekV3-score"] = "deepseek_score"
if "DeepseekV3-score" in df.columns:
    rename_dict["DeepseekV3-score"] = "deepseek_score"
df.rename(columns=rename_dict, inplace=True)

# Convert date column
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Define metrics and convert to numeric
metrics = [
    "token_similarity",
    "semantic_similarity",
    "exact_match",
    "gpt_score",
    "deepseek_score",
    "f1_ap",
    "f1_risk",
    "ap_factor_acc",
    "risk_consistency",
    "tara_validity"
]

# FIXED: Proper numeric conversion
for metric in metrics:
    if metric in df.columns:
        # Convert to string first to handle mixed types
        df[metric] = df[metric].apply(lambda x: pd.to_numeric(x, errors='coerce') )
        # Alternative: df[metric] = pd.to_numeric(df[metric], errors='coerce')

# Generate plots
for metric in metrics:
    if metric in df.columns:
        # Filter valid entries
        valid_rows = df.dropna(subset=['date', metric])
        
        if len(valid_rows) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(valid_rows['date'], valid_rows[metric], marker="o", label=metric)
            plt.title(f"{metric} über Trainingsläufe")
            plt.xlabel("Datum")
            plt.ylabel(metric)
            
            # Format date axis
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(Path(PLOT_DIR) / f"plot_{metric}.png")
            plt.close()

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        th { background-color: #f4f4f4; }
        tr:hover { background-color: #eef; }
        .high { background-color: #c8e6c9; }
        .low { background-color: #ffcdd2; }
        .notes { max-width: 300px; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>CVSS → Attack Potential Evaluation Report</h1>
    <p>Alle bisherigen Evaluationen mit Trainingsparametern und Qualitätsmetriken.</p>
    
    <table>
        <thead>
            <tr>
                {% for col in columns %}
                <th>{{ col }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in rows %}
            <tr>
                {% for col_name in columns %}
                    {% set val = row[loop.index0] %}
                <td class="
                    {% if col_name in score_columns and val != 'N/A' and val is number and val >= 0.8 %}high
                    {% elif col_name in score_columns and val != 'N/A' and val is number and val < 0.5 %}low
                    {% elif col_name in ['worked_best', 'worked_worst'] %}notes
                    {% endif %}
                ">
                    {{ val }}
                </td>
                
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <h2>Score-Entwicklung über Zeit</h2>
    {% for metric in metrics %}
        {% if metric in existing_metrics %}
        <h3>{{ metric }}</h3>
        <img src="plot_{{ metric }}.png" style="width:600px;">
        {% endif %}
    {% endfor %}
</body>
</html>
"""

# Prepare data for HTML table
df_display = df.fillna("N/A").copy()

# Convert datetime to string
if 'date' in df_display.columns:
    df_display['date'] = df_display['date'].apply(
        lambda x: x.strftime('%Y-%m-%d %H:%M') if not isinstance(x, str) and not pd.isna(x) else x
    )

columns = df_display.columns.tolist()
rows = df_display.values.tolist()
existing_metrics = [m for m in metrics if m in df.columns]

# Render HTML
template = Template(html_template)
html = template.render(
    columns=columns,
    rows=rows,
    metrics=metrics,
    existing_metrics=metrics,
    score_columns=metrics
)

with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ HTML-Report geschrieben nach: {HTML_OUTPUT}")