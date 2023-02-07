import json
import re
from collections import Counter
import glob
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def generate_color_palette(n_colors):
    color_palette = []
    for i in np.linspace(0, 1, n_colors):
        color = plt.cm.viridis(i)
        color_palette.append(color)
    return color_palette

if __name__ == '__main__':
    # model_path = "race_deberta_large_epoch_20"
    # exp_dirs = ["dpr_agent_dpr_sum_300_concat", "dpr_agent_pegasus_sum_300_concat", "extractive_dpr_agent"]
    #exp_dirs = ["dpr_agent_dpr_sum_300_concat", "dpr_agent_pegasus_sum_300_concat", "dpr_agent_pegasus_sum_rest_300_concat", "extractive_dpr_agent"]
    # exp_dirs = ["dpr_agent_dpr_sum_combine_20splits_maxlen_300_concat", "extractive_dpr_agent_first_20splits"]
    # exp_dirs = ["dpr_agent_dpr_sum_combine_20splits_maxlen_300_concat", "dpr_agent_dpr_sum_combine_20splits_maxlen_300_reverse_concat", "extractive_dpr_agent_first_20splits"]
    # exp_dirs = ["dpr_agent_dpr_sum_combine_20splits_maxlen_300_concat", "dpr_agent_dpr_sum_combine_20splits_maxlen_300_reverse_concat",
    #             "dpr_agent_dpr_sum_combine_20splits_maxlen_150_concat", "dpr_agent_dpr_sum_combine_20splits_maxlen_150_reverse_concat",
    #             "extractive_dpr_agent_first_20splits"][2:-1]
    # exp_dirs = [
    #     # model_path + "/dpr_agent_dpr_sum_combine_20splits_maxlen_150_25_concat",
    #     # model_path + "/dpr_agent_dpr_sum_combine_20splits_maxlen_150_50_concat",
    #     # model_path + "/dpr_agent_dpr_sum_combine_20splits_maxlen_150_100_concat",
    #     # model_path + "/dpr_agent_dpr_sum_combine_20splits_maxlen_150_concat",
    #     "t0_output/first_150_rest_25/T0pp",
    #     "t0_output/first_150_rest_50/T0pp",
    #     "t0_output/first_150_rest_100/T0pp",
    #     "t0_output/first_150_rest_150/T0pp",
    # ]
    filepath = "/data/chenghao/quality/baselines/experiment" + "/{}"
    fig = go.Figure()
    _exp_dirs = glob.glob(filepath.format("t0_improved_prompt_output/*/T0pp"))
    agent_dirs = [tuple(sorted([y for y in os.listdir(x) if "agent_" in y], key=lambda x: int(re.search("agent_(\d+)", x).group(1)) )) for x in _exp_dirs]
    # agent_dirs_num = [len(x) for x in agent_dirs]
    counter = Counter(agent_dirs)
    most_common_set_agent, most_common_agent_num = [x for x in counter.most_common() if len(x[0]) >= 4][0]

    exp_dirs = []
    for _dir, _agent_dirs in zip(_exp_dirs, agent_dirs):
        # if len(_agent_dirs) >= len(mo):
        if set(most_common_set_agent).issubset(set(_agent_dirs)):
            exp_dirs.append(_dir)

    # aliases = ["dpr-first-x% + dpr-rest-(1-x)%", "dpr-first-x% + pegasus-whole", "dpr-first-x%"]
    #aliases = ["dpr-first-x% + dpr-rest-(1-x)%", "dpr-first-x% + pegasus-whole", "dpr-first-x% + pegasus-rest-(1-x)%", "dpr-first-x%"]
    # aliases = ["dpr-first-x% + dpr-rest-(1-x)%", "dpr-first-x%"]
    # aliases = ["dpr-first-x% + dpr-rest-(1-x)%", "dpr-rest-(1-x)% + dpr-first-x%", "dpr-first-x%"]
    # aliases = ["dpr-dpr-300+300", "dpr-dpr-reverse-300+300", "dpr-dpr-150+150", "dpr-dpr-reverse-150+150", "dpr-first-x%-300"][2:-1]
    # aliases = ["DBT-dpr-dpr-150+25", "DBT-dpr-dpr-150+50", "DBT-dpr-dpr-150+100", "DBT-dpr-dpr-150+150"]
    # aliases += [
    #     "T0-dpr-150+25", "T0-dpr-150+50", "T0-dpr-150+100", "T0-dpr-150+150"
    # ]
    exp_dirs.sort(key=lambda x: x.split(os.sep)[-2])
    aliases = []
    for x in exp_dirs:
        name = x.split(os.sep)[-2]
        # print(name)
        _match = re.search("first_(?P<first>\d+)_rest_(?P<rest>\d+)", name)
        first, rest = _match.group("first"), _match.group("rest")
        aliases.append(f"T0-{first}/{rest}")

    # aliases = ["{}".format(x.split(os.sep)[-2]) for x in exp_dirs]
    # NUM_AGENTS = 5
    # NUM_AGENTS = 4
    NUM_AGENTS = 20
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["figure.figsize"] = (20, 6)
    # xs = [x*0.2 for x in list(range(NUM_AGENTS))]
    # xs = [x*0.05 for x in list(range(NUM_AGENTS))]
    xs = [x*(1/NUM_AGENTS) for x in sorted([int(re.search("agent_(\d+)", y).group(1)) for y in most_common_set_agent])]
    # proportions_str = [f"{x * 20}%" for x in list(range(NUM_AGENTS))]
    # proportions_str = [f"{x * 5}%" for x in list(range(NUM_AGENTS))]
    proportions_str = [f"{x * 5}%" for x in sorted([int(re.search("agent_(\d+)", y).group(1)) for y in most_common_set_agent])]
    # colors = ['r', "g", "purple", "b"]
    # colors = ['r', "g", "b", "purple", "orange", "black", "pink", ""]
    # colors = ['#7fc97f','#beaed4','#fdc086','black','#386cb0','#f0027f','#bf5b17','#666666']
    colors = [list(np.random.choice(range(256), size=3)) for _ in range(len(exp_dirs))]
    # colors = generate_color_palette(len(exp_dirs))
    # markers = [".", "o", "d", "*"]
    markers = [".", "o", "*", "d"] * (len(exp_dirs) // 4 + 1)
    # plt.xticks(xs, proportions_str)
    # plt.xlabel("First X% of QuALITY Context")
    # plt.ylabel("QA Accuracy")
    fig.update_xaxes(tickmode='array', tickvals=xs, ticktext=proportions_str)
    for exp_dir, alias, color, marker in zip(exp_dirs, aliases, colors, markers):
        buf = []
        for agent_i in most_common_set_agent:
            if "T0" not in exp_dir:
                agent_filepath = os.path.join(exp_dir, agent_i, "validation_metrics.json")
                performance_data = json.load(open(agent_filepath))
                performance = performance_data["eval_accuracy"]
                buf.append(performance)
            else:
                agent_filepath = os.path.join(exp_dir, agent_i, "results.json")
                performance_data = json.load(open(agent_filepath))
                performance = performance_data["evaluation"]["accuracy"]
                buf.append(performance)
        # plt.plot(xs, buf, label=alias, color=color, marker=marker)
        fig.add_trace(
            go.Scatter(x=xs, y=buf, name=alias, mode="lines", line=dict(color=f"rgb{tuple(color)}"))
        )
    # plt.axhline(y=45.58/100, label="T0pp-0-shot-full-document", color='orange', linestyle="--")
    fig.add_hline(y=45.58/100, annotation_text="T0pp-0-shot-full-document", line_color='orange', line_dash="dash")
    # plt.axhline(y=44.20/100, label="DBT-full-document", color='purple', linestyle="--")
    fig.add_hline(y=44.20/100, annotation_text="DBT-full-document", line_color='purple', line_dash="dash")
    # ax = plt.gca()
    # fig = plt.gcf()
    # box = ax.get_position()
    # # fig.set_size_inches(18.5, 10.5)
    # # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    # #                  box.width, box.height * 0.9])
    #
    # # Put a legend below current axis
    # # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    # #           fancybox=True, shadow=True, ncol=1, bbox_transform=fig.transFigure)
    # l3 = plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
    # fig.subplots_adjust(right=0.55)
    output_dir = "visualization"
    os.makedirs(output_dir, exist_ok=True)
    fig.update_layout(title="Agentwise QA-Accuracy Decomposition (T0pp-0-shot)",
                      font=dict(family="Arial", size=18, color="black"), xaxis_title="First X% of QuALITY Context", yaxis_title="QA Accuracy")
    fig.write_html(os.path.join(output_dir, "T0pp-improved-prompt-decomposition.html"))
    # # plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps-old3line.pdf"))
    # # plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps-20splits-rev.pdf"))
    # #plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps-20splits_150_150-rev.pdf"))
    # plt.savefig(os.path.join(output_dir, "with_t0_dpr-agent-prelim-exps-20splits_150_150-vary_length.pdf"))
    # # plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps.pdf"))
    # # plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps-20splits.pdf"))
    # plt.show()
