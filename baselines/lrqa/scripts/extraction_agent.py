import argparse
import os
import lrqa.preproc.extraction_agent as extraction
import lrqa.utils.io_utils as io

PHASES = ["train", "validation", "test"]


def get_scorer(scorer_name, args):
    if scorer_name == "rouge":
        return extraction.SimpleScorer()
    elif scorer_name == "fasttext":
        return extraction.FastTextScorer(extraction.load_fasttext_vectors(
            fname=args.fasttext_path,
            max_lines=100_000,
        ))
    elif scorer_name == "dpr":
        return extraction.DPRScorer(context_encoder_name=args.dpr_scorer_ctx_encoder,
                                    question_encoder_name=args.dpr_scorer_question_encoder,
                                    tokenizer_name=args.dpr_scorer_tokenizer, device="cuda:0")
    else:
        raise KeyError(scorer_name)


def main():
    parser = argparse.ArgumentParser(description="Do extractive preprocessing")
    parser.add_argument("--input_base_path", type=str, required=True,
                        help="Path to folder of cleaned inputs")
    parser.add_argument("--output_base_path", type=str, required=True,
                        help="Path to write processed outputs to")
    parser.add_argument("--scorer", type=str, default="rouge",
                        help="{rouge, fasttext, dpr}")
    parser.add_argument("--dpr_scorer_ctx_encoder", type=str, default="facebook/dpr-ctx_encoder-multiset-base",
                        help="dpr_scorer_ctx_encoder_path")
    parser.add_argument("--dpr_scorer_question_encoder", type=str,
                        default="facebook/dpr-question_encoder-multiset-base",
                        help="dpr_scorer_question_encoder_path")
    parser.add_argument("--dpr_scorer_tokenizer", type=str, default="facebook/dpr-question_encoder-multiset-base",
                        help="dpr_scorer_tokenizer")
    parser.add_argument("--agent_ids", type=str, default=",".join([str(x) for x in range(20)]),
                        help="specific agent_ids")
    parser.add_argument("--query_type", type=str, default="question",
                        help="{question, oracle_answer, oracle_question_answer}")
    parser.add_argument("--fasttext_path", type=str, default="/path/to/crawl-300d-2M.vec",
                        help="Pickle of fasttext vectors. (Only used for fasttext.)")
    args = parser.parse_args()
    os.makedirs(args.output_base_path, exist_ok=True)
    scorer = get_scorer(scorer_name=args.scorer, args=args)

    for phase in PHASES:
        extraction.process_file(
            input_path=os.path.join(args.input_base_path, f"{phase}.jsonl"),
            output_path=os.path.join(args.output_base_path, f"{phase}.jsonl"),
            scorer=scorer,
            query_type=args.query_type,
            verbose=True,
            original_article=False,
            agent_ids=args.agent_ids
        )

    io.write_json(
        {"num_choices": 4},
        os.path.join(args.output_base_path, "config.json"),
    )


if __name__ == "__main__":
    main()
