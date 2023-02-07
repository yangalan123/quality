import json
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_path = "race_deberta_large_epoch_20"
    # exp_dirs = ["dpr_agent_dpr_sum_300_concat", "dpr_agent_pegasus_sum_300_concat", "extractive_dpr_agent"]
    #exp_dirs = ["dpr_agent_dpr_sum_300_concat", "dpr_agent_pegasus_sum_300_concat", "dpr_agent_pegasus_sum_rest_300_concat", "extractive_dpr_agent"]
    # exp_dirs = ["dpr_agent_dpr_sum_combine_20splits_maxlen_300_concat", "extractive_dpr_agent_first_20splits"]
    # exp_dirs = ["dpr_agent_dpr_sum_combine_20splits_maxlen_300_concat", "dpr_agent_dpr_sum_combine_20splits_maxlen_300_reverse_concat", "extractive_dpr_agent_first_20splits"]
    # exp_dirs = ["dpr_agent_dpr_sum_combine_20splits_maxlen_300_concat", "dpr_agent_dpr_sum_combine_20splits_maxlen_300_reverse_concat",
    #             "dpr_agent_dpr_sum_combine_20splits_maxlen_150_concat", "dpr_agent_dpr_sum_combine_20splits_maxlen_150_reverse_concat",
    #             "extractive_dpr_agent_first_20splits"][2:-1]
    exp_dirs = [
        model_path + "/dpr_agent_dpr_sum_combine_20splits_maxlen_150_25_concat",
        model_path + "/dpr_agent_dpr_sum_combine_20splits_maxlen_150_50_concat",
        model_path + "/dpr_agent_dpr_sum_combine_20splits_maxlen_150_100_concat",
        model_path + "/dpr_agent_dpr_sum_combine_20splits_maxlen_150_concat",
        "t0_output/first_150_rest_25/T0pp",
        "t0_output/first_150_rest_50/T0pp",
        "t0_output/first_150_rest_100/T0pp",
        "t0_output/first_150_rest_150/T0pp",
    ]
    # aliases = ["dpr-first-x% + dpr-rest-(1-x)%", "dpr-first-x% + pegasus-whole", "dpr-first-x%"]
    #aliases = ["dpr-first-x% + dpr-rest-(1-x)%", "dpr-first-x% + pegasus-whole", "dpr-first-x% + pegasus-rest-(1-x)%", "dpr-first-x%"]
    # aliases = ["dpr-first-x% + dpr-rest-(1-x)%", "dpr-first-x%"]
    # aliases = ["dpr-first-x% + dpr-rest-(1-x)%", "dpr-rest-(1-x)% + dpr-first-x%", "dpr-first-x%"]
    # aliases = ["dpr-dpr-300+300", "dpr-dpr-reverse-300+300", "dpr-dpr-150+150", "dpr-dpr-reverse-150+150", "dpr-first-x%-300"][2:-1]
    aliases = ["DBT-dpr-dpr-150+25", "DBT-dpr-dpr-150+50", "DBT-dpr-dpr-150+100", "DBT-dpr-dpr-150+150"]
    aliases += [
        "T0-dpr-150+25", "T0-dpr-150+50", "T0-dpr-150+100", "T0-dpr-150+150"
    ]
    filepath = "/data/chenghao/quality/baselines/experiment" + "/{}"
    # NUM_AGENTS = 5
    # NUM_AGENTS = 4
    NUM_AGENTS = 20
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["figure.figsize"] = (20, 6)
    # xs = [x*0.2 for x in list(range(NUM_AGENTS))]
    xs = [x*0.05 for x in list(range(NUM_AGENTS))]
    # proportions_str = [f"{x * 20}%" for x in list(range(NUM_AGENTS))]
    proportions_str = [f"{x * 5}%" for x in list(range(NUM_AGENTS))]
    # colors = ['r', "g", "purple", "b"]
    # colors = ['r', "g", "b", "purple", "orange", "black", "pink", ""]
    colors = ['#7fc97f','#beaed4','#fdc086','black','#386cb0','#f0027f','#bf5b17','#666666']
    # markers = [".", "o", "d", "*"]
    markers = [".", "o", "*", "d"] * 2
    plt.xticks(xs, proportions_str)
    plt.xlabel("First X% of QuALITY Context")
    plt.ylabel("QA Accuracy")
    for exp_dir, alias, color, marker in zip(exp_dirs, aliases, colors, markers):
        buf = []
        for agent_i in range(NUM_AGENTS):
            if "T0" not in exp_dir:
                agent_filepath = os.path.join(filepath.format(exp_dir), f"agent_{agent_i}", "validation_metrics.json")
                performance_data = json.load(open(agent_filepath))
                performance = performance_data["eval_accuracy"]
                buf.append(performance)
            else:
                agent_filepath = os.path.join(filepath.format(exp_dir), f"agent_{agent_i}", "results.json")
                performance_data = json.load(open(agent_filepath))
                performance = performance_data["evaluation"]["accuracy"]
                buf.append(performance)
        plt.plot(xs, buf, label=alias, color=color, marker=marker)
    plt.axhline(y=45.58/100, label="T0pp-0-shot-full-document", color='orange', linestyle="--")
    plt.axhline(y=44.20/100, label="DBT-full-document", color='purple', linestyle="--")
    ax = plt.gca()
    fig = plt.gcf()
    box = ax.get_position()
    # fig.set_size_inches(18.5, 10.5)
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=1, bbox_transform=fig.transFigure)
    l3 = plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
    # fig.subplots_adjust(right=0.55)
    output_dir = "visualization"
    os.makedirs(output_dir, exist_ok=True)
    # plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps-old3line.pdf"))
    # plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps-20splits-rev.pdf"))
    #plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps-20splits_150_150-rev.pdf"))
    plt.savefig(os.path.join(output_dir, "with_t0_dpr-agent-prelim-exps-20splits_150_150-vary_length.pdf"))
    # plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps.pdf"))
    # plt.savefig(os.path.join(output_dir, "dpr-agent-prelim-exps-20splits.pdf"))
    plt.show()
