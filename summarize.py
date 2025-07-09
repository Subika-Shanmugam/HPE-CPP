import os
import re
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

COLUMN_DESCRIPTIONS = {
    'queueLength': "Number of requests waiting in the system queue.",
    'latencyRead': "Time taken to read data (in milliseconds).",
    'latencyWrite': "Time taken to write data (in milliseconds).",
    'iopsRead': "Input/Output operations per second for read operations.",
    'iopsWrite': "Input/Output operations per second for write operations.",
    'throughputRead': "Amount of data read per second (in MB/s).",
    'throughputWrite': "Amount of data written per second (in MB/s)."
}


def calculate_statistics(df):
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    return stats


def format_stats(stats):
    text = ""
    for col, metrics in stats.items():
        text += f"**{col}**:\n"
        for stat, value in metrics.items():
            text += f"  - {stat}: {value:.2f}\n"
    return text


def generate_prompt(df, stats):
    column_description_text = "\n".join([
        f"- **{col}**: {desc}" for col, desc in COLUMN_DESCRIPTIONS.items() if col in df.columns
    ])
    stats_text = format_stats(stats)

    prompt = f"""
You are a system performance analyst.

Below are column descriptions for the data:
{column_description_text}

Here are key computed statistics from the data:
{stats_text}

Strictly based on the above information:
- Summarize patterns, bottlenecks, or anomalies in 5 to 7 sentences.
- Do NOT assume anything beyond the data.
- Be objective, concise, and data-driven.
"""
    return prompt.strip()


def clean_filename(text):
    return re.sub(r'[^\w\-_.]', '_', text)


def plot_metrics(df, csv_name):
    base_name = os.path.splitext(os.path.basename(csv_name))[0]
    week_folder = os.path.join('plots', base_name)
    os.makedirs(week_folder, exist_ok=True)

    df = df.reset_index(drop=True)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure()
            df[col].plot(kind='line', marker='o', title=f"{col} over Time")
            plt.xlabel("Time (Index)")
            plt.ylabel(col)
            plt.grid(True)
            plt.tight_layout()

            safe_col = clean_filename(col)
            filename = os.path.join(week_folder, f"{safe_col}.png")
            plt.savefig(filename)
            print(f" Saved plot: {filename}")
            plt.close()


def generate_summary(csv_file, output_txt):
    df = pd.read_csv(csv_file).dropna().head(100)
    stats = calculate_statistics(df)
    prompt = generate_prompt(df, stats)
    plot_metrics(df, csv_file)

    result = subprocess.run(
        ['ollama', 'run', 'mistral'],
        input=prompt.encode('utf-8'),
        capture_output=True
    )

    if result.returncode != 0:
        print(f" Error while running ollama for {csv_file}")
        print(result.stderr.decode('utf-8'))
    else:
        summary = result.stdout.decode('utf-8').strip()
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f" Summary written to {output_txt}")

def compare_summaries(summary_files, output_file):
    comparisons = []
    for i in range(len(summary_files) - 1):
        with open(summary_files[i], 'r', encoding='utf-8') as f1, open(summary_files[i+1], 'r', encoding='utf-8') as f2:
            summary1 = f1.read().strip()
            summary2 = f2.read().strip()

            prompt = f"""
You are a system performance analyst. Below are two weekly summaries of system metrics.

Compare Week {i+1} and Week {i+2}:
- Focus only on actual differences in trends or anomalies.
- Do not repeat similar parts.
- Be precise and analytical.

--- Week {i+1} Summary ---
{summary1}

--- Week {i+2} Summary ---
{summary2}
""".strip()

            result = subprocess.run(
                ['ollama', 'run', 'mistral'],  
                input=prompt.encode('utf-8'),
                capture_output=True
            )

            if result.returncode != 0:
                print(f" Error comparing summaries for Week {i+1} and Week {i+2}")
                print(result.stderr.decode('utf-8'))
                continue

            comparison_text = result.stdout.decode('utf-8').strip()
            comparisons.append(f"\n🔍 Week {i+1} vs Week {i+2}:\n{comparison_text}\n{'-'*80}\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(comparisons)
    print(f" Differences saved to {output_file}")

# Main runner
if __name__ == "__main__":
    os.makedirs('summaries', exist_ok=True)
    os.makedirs('comparisons', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    csv_files = ['week1.csv', 'week2.csv', 'week3.csv', 'week4.csv']
    summary_files = [f'summaries/week{i+1}_summary.txt' for i in range(4)]

    for csv, summary in zip(csv_files, summary_files):
        generate_summary(csv, summary)

    compare_summaries(summary_files, 'comparisons/summary_differences.txt')
