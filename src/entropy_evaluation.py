import os
import numpy as np
import pandas as pd
import json
import seaborn as sns
import plotly.graph_objects as go
from nltk.corpus import stopwords 
import string
from collections import defaultdict
import matplotlib.pyplot as plt

STOPWORDS = set(stopwords.words("english"))
PUNCTUATION = set(string.punctuation)

def zscore_entropy_analysis(model_outputs):
    """
    Compute per-token entropy statistics including delta and z-score.
    """
    per_step = model_outputs["per_step"]
    avg = model_outputs["average_entropy"]
    entropies = [s["entropy"] for s in per_step]
    std = np.std(entropies)
    analyzed = []
    for step in per_step:
        e = step["entropy"]
        z = (e - avg) / (std + 1e-9)
        analyzed.append({"token": step["generated_token"],"z_score": z})
    return analyzed


def load_mfd(filepath):
    """
    Load MFD 2.0.
    Return dictionary {foundation: set of words}.
    """
    with open(filepath, "r") as f:
        content = f.read()

    sections = content.split("%")
    code_to_foundation = {}

    # First section: code and foundation name
    for line in sections[1].split("\n"):
        line = line.strip()
        if line:
            code, name = line.split("\t")
            code_to_foundation[code.strip()] = name.strip()

    mfd_dict = {name: set() for name in code_to_foundation.values()}

    for line in sections[2].split("\n"):
        line = line.strip()
        if line:
            word, code = line.split("\t")
            word = word.lower()
            code = code.strip()
            if code in code_to_foundation:
                mfd_dict[code_to_foundation[code]].add(word)
    return mfd_dict


def combine_lexicons(mfd_path="../data/mfd2.0.dic", emfd_path="../data/emfd_single_vice_virtue.pkl"):
    """
    Combine MFD 2.0 and EMFD lexicons. Return dictionary {word: {foundation: foundation, "score": score}}.
    """
    mfd2_dict= load_mfd(mfd_path)
    emfd_dict = pd.read_pickle(emfd_path)
    combined_lexicons = emfd_dict.copy()

    for foundation, words in mfd2_dict.items():
        for word in words:
            if word not in combined_lexicons:
                combined_lexicons[word] = {'foundation': foundation, 'score': None}
    return combined_lexicons
 

def load_and_compute_zscore(model_output_dir):
    """
    Load model outputs from a directory and compute z-score entropy analysis for each token.
    """
    models = ['Apertus-8B-Instruct-2509',
              'Llama-3.1-8B-Instruct',
                'llama-3.2-1B-Instruct',
                'Llama-3.2-3B-Instruct',
                'Llama-3.1-70B-Instruct',
                'Qwen2.5-7B-Instruct','Qwen2.5-14B-Instruct', 'Qwen2.5-72B-Instruct',]
    files = {}

    for model_dir in models:
        path = os.path.join(model_output_dir, model_dir)
        files[model_dir] = {}  
        for fname in os.listdir(path):
            topic = fname.split("_")[-1].replace(".json", "")
            with open(os.path.join(path, fname)) as f:
                obj = json.load(f)
                files[model_dir][topic] = obj['generations']

    prompt_conditions = ['base','care','equality','proportionality','loyalty','authority','purity']
    data = []

    for model in models:
        for topic in files[model]:
            generations = files[model][topic]  
            for gen in generations:  
                for prompt_condition in prompt_conditions:
                    replies = gen['replies'].get(prompt_condition)
                    if replies and "entropy" in replies:
                        analyzed_list = zscore_entropy_analysis(replies["entropy"])
                        analyzed_df = pd.DataFrame(analyzed_list)
                        analyzed_df["topic"] = topic
                        analyzed_df["prompt_condition"] = prompt_condition
                        analyzed_df["model"] = model
                        data.append(analyzed_df)

    df = pd.concat(data, ignore_index=True)
    df['model'] = df['model'].replace({'llama-3.2-1B-Instruct': 'Llama-3.2-1B-Instruct', })
    return df


def moral_annotation(df, combined_lexicon):
    """
    Annotate tokens with moral foundation.
    """
    rules = [
        ('inging', 'ing'), ('inging', ''),
        ('tion',  ''),  ('sses', 'ss'), ('ies', 'y'), ('ied', 'y'),
        ('ing',   ''),  ('ings', ''),   ('edly', ''), ('ness', ''),
        ('ses',   's'), ('zes',  'z'),  ('xes',  'x'),('ches', 'ch'),
        ('shes',  'sh'),('ves',  'f'),  ('ves',  've'),('s',   ''),
        ('ed',    ''),  ('ed',   'e'),  ('er',   ''),  ('est', ''),
        ('ly',    ''),  ('al',   ''),   ('ial',  ''),]

    dict_lemma_map = {}
    for key in combined_lexicon.keys():
        for candidate in get_candidates(key, rules):
            dict_lemma_map[candidate] = key

    def tag_token(token):
        for candidate in get_candidates(str(token), rules):
            if candidate in dict_lemma_map:
                matched_key = dict_lemma_map[candidate]
                return combined_lexicon[matched_key]['foundation'], matched_key
        return None, None

    df['moral'], df['matched_word'] = zip(*df['token'].apply(tag_token))
    df['polarity'] = df['moral'].str.split(".").str[1]
    return df

def get_candidates(word, rules):
    """
    Auxiliary function to generate candidate lemmas for a given word based on suffix replacement rules.
    """
    word = word.lower().strip()
    candidates = {word}
    for suffix, replacement in rules:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: len(word) - len(suffix)] + replacement
            candidates.add(stem)
            if len(stem) >= 4 and stem[-1] == stem[-2]:
                candidates.add(stem[:-1])
    return candidates


#-----------------PLOTS------------
def plot_tokens_per_topic_model(df, name):
    """
    Plots the number of tokens per topic and model, grouped by prompt condition.
    """
    for p in sorted(df["prompt_condition"].unique()):
        filtered_df = df[df["prompt_condition"] == p]

        tokens_por_topic_model = (
            filtered_df
            .groupby(["model", "topic"])["token"]
            .count()
            .reset_index(name="n_tokens"))

        pivot = tokens_por_topic_model.pivot(
            index="topic",
            columns="model",
            values="n_tokens").fillna(0)

        pivot = pivot.reindex(sorted(df["topic"].unique()))
        pivot = pivot[sorted(df["model"].unique())]
        fig = go.Figure()

        for model in sorted(df["model"].unique()):
            fig.add_trace(
                go.Bar(
                    x=pivot.index,
                    y=pivot[model],
                    name=model,
                    text=pivot[model],
                    textposition="outside"))

        fig.update_layout(
            barmode="group",
            title=f"Prompt statements — {p}",
            xaxis_title=None,
            yaxis_title="Number of tokens",
            xaxis_tickangle=90,
            height=500)

        fig.write_html(f"../charts/entropy/{name}_{p}.html")
        fig.show()


def plot_virtue_vice(df, name):
    """
    Plots the percentage of virtue and vice tokens per topic and model.
    """
    virtue_table = (
        df.groupby(["topic", "model"])
        .apply(virtue_pct)
        .unstack()
        .reindex(index=sorted(df["topic"].unique()))
        .fillna(0))
    virtue_table["Mean"] = virtue_table.mean(axis=1)
    
    vice_table = (
        df.groupby(["topic", "model"])
        .apply(vice_pct)
        .unstack()
        .reindex(index=sorted(df["topic"].unique()))
        .fillna(0))
    vice_table["Mean"] = vice_table.mean(axis=1)

    virtue_table.index.name = None
    virtue_table.columns.name = None
    vice_table.index.name = None
    vice_table.columns.name = None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    sns.heatmap(virtue_table, ax=ax1, annot=True, fmt=".2f",
                cmap="YlGn", vmin=0, vmax=1, linewidths=0.3,
                #cbar_kws={"label": "% virtue"}
                )
    col = -1
    for idx in np.arange(virtue_table.size).reshape(virtue_table.shape)[:, col]:
        ax1.texts[idx].set_fontweight('bold')
    ax1.set_title("% Virtue", fontsize=9, fontweight="bold")
    ax1.tick_params(axis="x", rotation=90, labelsize=7)
    ax1.tick_params(axis="y", labelsize=8, length=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), fontweight="bold")

    sns.heatmap(vice_table, ax=ax2, annot=True, fmt=".2f",
                cmap="OrRd", vmin=0, vmax=1, linewidths=0.3,
                #cbar_kws={"label": "% vice"}
                )
    col = -1
    for idx in np.arange(vice_table.size).reshape(vice_table.shape)[:, col]:
        ax2.texts[idx].set_fontweight('bold')
    ax2.set_title("% Vice", fontsize=9, fontweight="bold")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=90, labelsize=7)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"../charts/entropy/{name}.pdf", format="pdf", bbox_inches="tight")

    result_ = pd.concat((virtue_table["Mean"].round(2), vice_table["Mean"].round(2)), axis=1)
    print(result_.to_latex())
    return result_
    # plt.show()


def plot_virtue_vice_variance(df, name):
    """
    Plots the variance of virtue/vice % across prompt conditions
    for each (topic, model) pair as dual heatmaps.
    """
    # Compute virtue% per (topic, model, prompt_condition)
    virtue_per_cond = (
        df.groupby(["topic", "model", "prompt_condition"])
        .apply(virtue_pct)
        .reset_index(name="virtue_pct")
    )
    virtue_var_table = (
        virtue_per_cond.pivot_table(
            index="topic", columns="model",
            values="virtue_pct", aggfunc="var")
        .reindex(index=sorted(df["topic"].unique()))
        .fillna(0)
    )
    virtue_var_table["Var"] = virtue_var_table.mean(axis=1)

    # Compute vice% per (topic, model, prompt_condition)
    vice_per_cond = (
        df.groupby(["topic", "model", "prompt_condition"])
        .apply(vice_pct)
        .reset_index(name="vice_pct")
    )
    vice_var_table = (
        vice_per_cond.pivot_table(
            index="topic", columns="model",
            values="vice_pct", aggfunc="var")
        .reindex(index=sorted(df["topic"].unique()))
        .fillna(0)
    )
    vice_var_table["Mean"] = vice_var_table.mean(axis=1)

    # Clean index/column names
    virtue_var_table.index.name = None
    virtue_var_table.columns.name = None
    vice_var_table.index.name = None
    vice_var_table.columns.name = None

    # Plot dual heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    sns.heatmap(virtue_var_table, ax=ax1, annot=True, fmt=".3f",
                cmap="YlGn", linewidths=0.3)
    col = -1
    for idx in np.arange(virtue_var_table.size).reshape(virtue_var_table.shape)[:, col]:
        ax1.texts[idx].set_fontweight('bold')
    ax1.set_title("Var(% Virtue) across prompt conditions", fontsize=9, fontweight="bold")
    ax1.tick_params(axis="x", rotation=90, labelsize=7)
    ax1.tick_params(axis="y", labelsize=8, length=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), fontweight="bold")

    sns.heatmap(vice_var_table, ax=ax2, annot=True, fmt=".3f",
                cmap="OrRd", linewidths=0.3)
    col = -1
    for idx in np.arange(vice_var_table.size).reshape(vice_var_table.shape)[:, col]:
        ax2.texts[idx].set_fontweight('bold')
    ax2.set_title("Var(% Vice) across prompt conditions", fontsize=9, fontweight="bold")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=90, labelsize=7)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"../charts/entropy/{name}.pdf", format="pdf", bbox_inches="tight")

    result_ = virtue_var_table["Var"].round(4)
    print(result_.to_latex())
    return result_


def plot_virtue_vice_by_foundation(df, name):
    """
    Plots the percentage of virtue and vice tokens per moral foundation and model.
    """
    # Filter to only moral tokens and extract foundation
    df_moral = df[df["polarity"].isin(["virtue", "vice"])].copy()
    df_moral["foundation"] = df_moral["moral"].str.split(".").str[0]

    # Compute virtue% per (foundation, model)
    virtue_table = (
        df_moral.groupby(["foundation", "model"])
        .apply(lambda g: (g["polarity"] == "virtue").sum() / len(g))
        .unstack()
        .fillna(0)
    )
    virtue_table["Mean"] = virtue_table.mean(axis=1)
    virtue_table.loc["Mean", :] = virtue_table.mean(axis=0)

    # Compute vice% per (foundation, model)
    vice_table = (
        df_moral.groupby(["foundation", "model"])
        .apply(lambda g: (g["polarity"] == "vice").sum() / len(g))
        .unstack()
        .fillna(0)
    )
    vice_table["Mean"] = vice_table.mean(axis=1)
    vice_table.loc["Mean", :] = vice_table.mean(axis=0)

    # Clean index/column names
    virtue_table.index.name = None
    virtue_table.columns.name = None
    vice_table.index.name = None
    vice_table.columns.name = None

    # Plot dual heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    sns.heatmap(virtue_table, ax=ax1, annot=True, fmt=".2f",
                cmap="YlGn", vmin=0, vmax=1, linewidths=0.3)
    col = -1
    for idx in np.arange(virtue_table.size).reshape(virtue_table.shape)[:, col]:
        ax1.texts[idx].set_fontweight('bold')
    row = -1
    for idx in np.arange(virtue_table.size).reshape(virtue_table.shape)[row, :]:
        ax1.texts[idx].set_fontweight('bold')
    ax1.set_title("% Virtue", fontsize=9, fontweight="bold")
    ax1.tick_params(axis="x", rotation=90, labelsize=7)
    ax1.tick_params(axis="y", labelsize=8, length=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), fontweight="bold")

    sns.heatmap(vice_table, ax=ax2, annot=True, fmt=".2f",
                cmap="OrRd", vmin=0, vmax=1, linewidths=0.3)
    col = -1
    for idx in np.arange(vice_table.size).reshape(vice_table.shape)[:, col]:
        ax2.texts[idx].set_fontweight('bold')
    row = -1
    for idx in np.arange(vice_table.size).reshape(vice_table.shape)[row, :]:
        ax2.texts[idx].set_fontweight('bold')
    ax2.set_title("% Vice", fontsize=9, fontweight="bold")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=90, labelsize=7)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"../charts/entropy/{name}.pdf", format="pdf", bbox_inches="tight")

    result_ = pd.concat((virtue_table["Mean"].round(2), vice_table["Mean"].round(2)), axis=1)
    print(result_.to_latex())
    return result_


def virtue_pct(g):
    moral = g[g["polarity"].isin(["virtue", "vice"])]
    total = len(moral)
    return (moral["polarity"] == "virtue").sum() / total if total > 0 else 0

def vice_pct(g):
    moral = g[g["polarity"].isin(["virtue", "vice"])]
    total = len(moral)
    return (moral["polarity"] == "vice").sum() / total if total > 0 else 0


def plot_virtue_vice_by_prompt_condition(df, name):
    """
    Plots the percentage of virtue and vice tokens per topic and model, grouped by prompt condition.
    """
    prompt_conditions = sorted(df["prompt_condition"].unique())
    for model in sorted(df["model"].unique()):
        df_m = df[df["model"] == model]
        
        fig, axes = plt.subplots(
            1, len(prompt_conditions),
            figsize=(14, 8),
            sharey=True)
        
        if len(prompt_conditions) == 1:
            axes = [axes]
        
        for ax, prompt in zip(axes, prompt_conditions):
            df_mp = df_m[df_m["prompt_condition"] == prompt]
            
            if df_mp.empty:
                ax.set_visible(False)
                continue
            
            table = pd.crosstab(
                df_mp["topic"],
                df_mp["polarity"],
                normalize="index"         
            ).reindex(columns=["virtue", "vice"], fill_value=0)

            sns.heatmap(
                table,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="YlOrBr",            
                vmin=0, vmax=1,
                cbar=ax == axes[-1],       
                linewidths=0.3,
                annot_kws={"size": 6})
            
            ax.set_title(prompt, fontsize=7, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="y", length=0)

            for label in ax.get_xticklabels():
                label.set_fontweight("bold")
            for label in ax.get_yticklabels():
                label.set_fontweight("bold")
     
        fig.suptitle(f"{model}", fontsize=10, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(f"../charts/entropy/{name}_{model}.pdf",format="pdf", bbox_inches="tight")
        plt.show()

def plot_virtue_vice_distance_to_base_prompt(df, name):
    crosstabs = defaultdict(dict)

    prompt_conditions = sorted(df["prompt_condition"].unique())
    model = "Apertus-8B-Instruct-2509"
    for model in sorted(df["model"].unique()):
        crosstabs[model] = defaultdict(dict)
        df_m = df[df["model"] == model]
        prompt = "base"
        for prompt in prompt_conditions:
            df_mp = df_m[df_m["prompt_condition"] == prompt]

            crosstabs[model][prompt] = pd.crosstab(
                df_mp["topic"],
                df_mp["polarity"],
                normalize="index",    
            ).reindex(columns=["virtue", "vice"], fill_value=0)

    prompt_conditions = sorted(df["prompt_condition"].unique())
    prompt_conditions.remove("base")

    differences = defaultdict(dict)
    for model in sorted(df["model"].unique()):
        differences[model] = defaultdict(dict)
        for prompt in prompt_conditions:
            differences[model][prompt] = np.abs(crosstabs[model][prompt]["virtue"] - crosstabs[model]["base"]["virtue"])

    differences = pd.concat(
        {outer: pd.DataFrame(inner) for outer, inner in differences.items()},
        axis=1
    )
    differences.columns.names = ["model", "moral"]
    differences.loc["Mean", :] = differences.mean(axis=0)

    table = differences.loc["Mean"].reset_index().pivot(index="moral", columns="model", values="Mean")
    table["Mean"] = table.mean(axis=1)
    table.loc["Mean", :] = table.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(
        table,
        annot=True,
        fmt=".2f",
        cmap="YlGn",            
        vmin=0, vmax=1,
        linewidths=0.3,
        annot_kws={"size": 10})
    col = -1
    for idx in np.arange(table.size).reshape(table.shape)[:, col]:
        ax.texts[idx].set_fontweight('bold')
    row = -1
    for idx in np.arange(table.size).reshape(table.shape)[row, :]:
        ax.texts[idx].set_fontweight('bold')
    plt.title("Distance to base prompt", fontsize=15)
    plt.savefig(f"../charts/entropy/.pdf", format="pdf", bbox_inches="tight")



#----- MAIN------------

#load lexicons, generations and compute zscore
df = load_and_compute_zscore("../output/generations")
lexicon = combine_lexicons()

#plot all tokens per topic and model
plot_tokens_per_topic_model(df,"all_tokens_per_topic_model")

#filter high z-score tokens and plot tokens per topic and model
threshold = 2.0
df= df[df['z_score'] > threshold]
df = df[~df['token'].str.strip().str.lower().isin(STOPWORDS)]
df = df[~df['token'].str.strip().str.lower().isin(PUNCTUATION)]
df= df[df['z_score'] > threshold]
plot_tokens_per_topic_model(df,"high_zscore_tokens_per_topic_model")

#annotate moral foundation and plot virtue/vice distribution
df = moral_annotation(df, combined_lexicon=lexicon)
result_ = plot_virtue_vice(df,'high_zscore_tokens_moral_distribution_virtue_vice')
result_variance_ = plot_virtue_vice_variance(df, 'high_zscore_variance_virtue_vice_by_prompt_condition')
with open("../charts/entropy/table.txt", "w") as f:
    result_ = pd.merge(result_, result_variance_, left_index=True, right_index=True).\
        round({"Mean": 2, "Var": 3}).sort_values("Var", ascending=False).to_latex()
    f.write(result_)
plot_virtue_vice_by_foundation(df, 'high_zscore_moral_distribution_by_foundation')
plot_virtue_vice_distance_to_base_prompt(df, "high_zcore_moral_distance_to_base_prompt")
plot_virtue_vice_by_prompt_condition(df,'high_zscore_tokens_moral_distribution_by_prompt_condition')


