import json
import os
import sys
import time
from typing import Callable, Union

from src import crawler, cleaner, detector, fixer, parser, annotator, reviewer, util, api_openai, api_ollama
from src.api_ollama import ApiOllama
from src.api_openai import ApiOpenAI


def prepare_batch(run_id: str, task: str, in_folder: str, func: Callable[[str, str, str], list[dict]]):
    batch = []
    for filename in os.listdir(in_folder):
        # Ensure we are processing files only (not directories)
        if os.path.isfile(os.path.join(in_folder, filename)):
            # Extract the package name from the filename
            pkg = os.path.splitext(filename)[0]
            entries = func(run_id, pkg, in_folder)
            if entries is not None:
                batch += entries

    jsonl = ''
    for entry in batch:
        jsonl += json.dumps(entry) + '\n'

    util.write_to_file(f"output/{run_id}/batch/{task}/batch_input.jsonl", jsonl)


def run_batch(run_id: str, task: str, in_folder: str, client: Union[ApiOpenAI, ApiOllama], model: str, func: Callable[[str, str, str], list[dict]]):
    # prepare and run the batch
    prepare_batch(run_id, task, in_folder, func)
    client.run_batch(task)

    print(f"Batch {task} started. Waiting for completion...")

    # check the status of the batch every minute
    count = 0
    while True:
        count += 1
        batch_status = client.check_batch_status(task)
        if batch_status.status == "completed":
            print(f"Batch completed after {count} checks, fetching results...")
            client.get_batch_results(task)
            return True
        elif batch_status.status == "failed":
            print(f"Batch failed after {count} checks.")
            return False
        else:
            print(f"Batch still running after {count} checks...")
            time.sleep(60)


def run_pipeline(
        run_id: str,
        id_file: str = None,
        single_pkg: str = None,
        models: list[str] = None,
        client: Union[ApiOpenAI, ApiOllama] = None,
        crawl_retries: int = 2,
        skip_crawl: bool = True,
        skip_clean: bool = False,
        skip_detect: bool = False,
        skip_headline_fix: bool = False,
        skip_parse: bool = False,
        skip_annotate: bool = False,
        skip_review: bool = False,
        batch_detect: bool = False,
        batch_headline_fix: bool = False,
        batch_annotate: bool = False,
        batch_review: bool = False,
        parallel_prompt: bool = False
):
    # define the processing steps of the pipeline with the corresponding functions and parameters
    processing_steps = [
        ("clean", skip_clean, False, False, cleaner.execute, "original", "cleaned"),  # cleaner
        ("detector", not batch_detect, True, False, detector.prepare_batch, "cleaned", "accepted"),  # batch detector
        ("detect", skip_detect, False, batch_detect, detector.execute, "cleaned", "accepted"),  # detector
        ("fixer", not batch_headline_fix, True, False, fixer.prepare_batch, "accepted", "fixed"),  # batch fixer
        ("fix", skip_headline_fix, False, batch_headline_fix, fixer.execute, "accepted", "fixed"),  # fixer
        ("parse", skip_parse, False, False, parser.execute, "fixed", "json"),  # parser
        ("annotator", not batch_annotate, True, False, annotator.prepare_batch, "json", "annotated"),  # batch annotator
        ("annotate", skip_annotate, False, batch_annotate, annotator.execute, "json", "annotated"),  # annotator
        ("review", not batch_review, True, False, reviewer.prepare_batch, "annotated", "reviewed"),  # batch reviewer
        ("review", skip_review, False, batch_review, reviewer.execute, "annotated", "reviewed")  # reviewer
    ]

    # set up and configure the API client
    client.setup()

    # prepare the output folders for the given run id
    util.prepare_output(run_id, overwrite=False)

    # crawl the privacy policies of the given packages if required
    if not skip_crawl:
        out_folder = f"output/{run_id}/original"
        if single_pkg is not None:
            crawler.crawl_list([single_pkg], out_folder, crawl_retries)
        else:
            crawler.execute(id_file, out_folder, crawl_retries)

    # iterate over the processing steps and execute the corresponding function for each package
    for step, skip, is_batch_step, use_batch_result, func, in_folder, out_folder in processing_steps:
        in_folder = f"output/{run_id}/{in_folder}"
        out_folder = f"output/{run_id}/{out_folder}"

        model = None
        if models is not None:
            if models[step] is not None:
                model = models[step]

        if not skip:
            if single_pkg is not None:
                if is_batch_step:
                    # run the batch processing function of this step for the provided package
                    run_batch(run_id, step, in_folder, client, model, func)
                else:
                    # execute the processing function of this step for the provided package
                    func(run_id, single_pkg, in_folder, out_folder, client, model, use_batch_result, parallel_prompt)
            else:
                if is_batch_step:
                    # run the batch processing function of this step for all packages
                    run_batch(run_id, step, in_folder, client, model, func)
                else:
                    # iterate over all packages and execute the processing function of this step for each package
                    for filename in os.listdir(in_folder):
                        # ensure we are processing files only (not directories)
                        if os.path.isfile(os.path.join(in_folder, filename)):
                            # extract the package name from the filename
                            pkg = os.path.splitext(filename)[0]
                            try:
                                # execute the processing function of this step for the current package
                                func(run_id, pkg, in_folder, out_folder, client, model, use_batch_result, parallel_prompt)
                            except Exception:
                                with open(f"output/{run_id}/log/error.txt", 'w') as error_file:
                                    error_file.write(f"{pkg}\n")
        else:
            # copy the content of in_folder to out_folder without processing so that the next step can be executed
            for filename in os.listdir(in_folder):
                if os.path.isfile(os.path.join(in_folder, filename)):
                    with open(in_folder + '/' + filename, 'r') as file:
                        content = file.read()
                        with open(out_folder + '/' + filename, 'w') as new_file:
                            new_file.write(content)

    # pipeline done, close the API client
    client.close()


def parse_args():
    # parse the run id from the command line argument "-run-id" or exit if not provided
    run_id = None
    if "-run-id" in sys.argv:
        idx = sys.argv.index("-run-id")
        if idx + 1 < len(sys.argv):
            run_id = sys.argv[idx + 1]

    # parse the path to the file containing the package ids from the command line argument "-id-file"
    id_file = None
    if "-id-file" in sys.argv:
        idx = sys.argv.index("-id-file")
        if idx + 1 < len(sys.argv):
            id_file = sys.argv[idx + 1]

    # parse the single package from the command line argument "-pkg"
    single_pkg = None
    if "-pkg" in sys.argv:
        idx = sys.argv.index("-pkg")
        if idx + 1 < len(sys.argv):
            single_pkg = sys.argv[idx + 1]

    # parse the model from the command line argument "-model"
    model = None
    if "-model" in sys.argv:
        idx = sys.argv.index("-model")
        if idx + 1 < len(sys.argv):
            model = sys.argv[idx + 1]

    # parse the step-specific models from the command line arguments "-model-<step>"
    models = {
        "detect": None,
        "headline_fix": None,
        "annotate": None,
        "review": None
    }
    for step in models.keys():
        if f"-model-{step}" in sys.argv:
            idx = sys.argv.index(f"-model-{step}")
            if idx + 1 < len(sys.argv):
                models[step] = sys.argv[idx + 1]

    # parse whether to use parallel prompts from the command line argument "-parallel-prompt"
    parallel_prompt = "-parallel-prompt" in sys.argv

    # parse the number of crawl retries from the command line argument "-crawl-retries"
    crawl_retries = 2
    if "-crawl-retries" in sys.argv:
        idx = sys.argv.index("-crawl-retries")
        if idx + 1 < len(sys.argv):
            crawl_retries = int(sys.argv[idx + 1])

    # parse command line args that indicate which steps to skip
    skip_crawl = "-no-crawl" in sys.argv
    skip_clean = "-no-clean" in sys.argv
    skip_detect = "-no-detect" in sys.argv
    skip_headline_fix = "-no-headline-fix" in sys.argv
    skip_parse = "-no-parse" in sys.argv
    skip_annotate = "-no-annotate" in sys.argv
    skip_review = "-no-review" in sys.argv

    # parse command line args that indicate which LLM-reliant steps to run in batch mode (if supported by the client)
    batch_detect = "-batch-detect" in sys.argv
    batch_headline_fix = "-batch-headline-fix" in sys.argv
    batch_annotate = "-batch-annotate" in sys.argv
    batch_review = "-batch-review" in sys.argv

    return (
        run_id,
        id_file,
        single_pkg,
        model,
        models,
        crawl_retries,
        skip_crawl,
        skip_clean,
        skip_detect,
        skip_headline_fix,
        skip_parse,
        skip_annotate,
        skip_review,
        batch_detect,
        batch_headline_fix,
        batch_annotate,
        batch_review,
        parallel_prompt
    )


def parse_config_file(path: str):
    with open(path, 'r') as file:
        config = json.load(file)

    return (
        config.get("run_id", None),
        config.get("id_file", None),
        config.get("single_pkg", None),
        config.get("model", None),
        config.get("models", None),
        config.get("crawl_retries", 2),
        config.get("skip_crawl", True),
        config.get("skip_clean", False),
        config.get("skip_detect", False),
        config.get("skip_headline_fix", False),
        config.get("skip_parse", False),
        config.get("skip_annotate", False),
        config.get("skip_review", False),
        config.get("batch_detect", False),
        config.get("batch_headline_fix", False),
        config.get("batch_annotate", False),
        config.get("batch_review", False),
        config.get("parallel_prompt", False)
    )


def main():
    # parse the "-c" argument that indicates the path of a configuration file
    config_path = None
    if "-c" in sys.argv:
        idx = sys.argv.index("-c")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]

    # parse the command line arguments or the configuration file if provided
    if config_path is not None:
        run_id, id_file, single_pkg, model, models, crawl_retries, skip_crawl, skip_clean, skip_detect, skip_headline_fix, \
            skip_parse, skip_annotate, skip_review, batch_detect, batch_headline_fix, batch_annotate, batch_review, \
            parallel_prompt = parse_config_file(config_path)
    else:
        run_id, id_file, single_pkg, model, models, crawl_retries, skip_crawl, skip_clean, skip_detect, \
            skip_headline_fix, skip_parse, skip_annotate, skip_review, batch_detect, batch_headline_fix, batch_annotate, \
            batch_review, parallel_prompt = parse_args()

    # check whether a run_id is provided
    if run_id is None:
        print("Error: No run id provided. Please provide a run id with the -run-id argument or the run_id field "
              "in the configuration file.", file=sys.stderr)
        sys.exit(1)

    # check whether an id_file or single_pkg is provided
    if id_file is None and single_pkg is None:
        print("Error: No package ids provided. Please provide either a file with package ids with the -id-file "
              "argument or a single package id with the -pkg argument or the id_file or pkg field in the "
              "configuration file.", file=sys.stderr)
        sys.exit(1)

    # check whether a model or step-specific models are provided
    if model is None:
        for step in ["detect", "headline_fix", "annotate", "review"]:
            if models[step] is None:
                print(f"Error: No model provided for step {step}. Please provide a default model with the -model "
                      f"argument or the model field in the configuration file or specify a model for each pipeline "
                      f"step with the -model-<step> arguments or the models field in the configuration file.",
                      file=sys.stderr)
                sys.exit(1)

    # get the lists of valid models from the API clients
    openai_models = api_openai.models.keys()
    ollama_models = api_ollama.models.keys()

    # check whether the provided models are valid
    clients = []
    for model in models.values():
        if model not in openai_models and model not in ollama_models:
            print(f"Error: Invalid model provided: {model}. Please only provide valid models from the following list: "
                  f"{openai_models + ollama_models}", file=sys.stderr)
            sys.exit(1)
        elif model in openai_models:
            clients.append('openai')
        elif model in ollama_models:
            clients.append('ollama')

    # check whether the provided models are from the same API
    if not all(client == clients[0] for client in clients):
        print("Error: Models from different APIs are not supported. Please provide models from the same API.",
              file=sys.stderr)
        sys.exit(1)

    # create the API client
    if model is None:
        if clients[0] == 'openai':
            client = ApiOpenAI(run_id, model)
        else:
            client = ApiOllama(run_id, model)
    elif model in openai_models:
        client = ApiOpenAI(run_id, model)
    elif model in ollama_models:
        client = ApiOllama(run_id, model)
    else:
        print(f"Error: Invalid model provided: {model}. Please provide a valid model from the following list: "
              f"{openai_models + ollama_models}", file=sys.stderr)
        sys.exit(1)

    # for all batch steps, check whether the client supports batch processing
    batch_detect = batch_detect and client.supports_batch
    batch_headline_fix = batch_headline_fix and client.supports_batch
    batch_annotate = batch_annotate and client.supports_batch
    batch_review = batch_review and client.supports_batch

    # check whether the client supports parallel prompts
    parallel_prompt = parallel_prompt and client.supports_parallel

    run_pipeline(run_id=run_id,
                 id_file=id_file,
                 single_pkg=single_pkg,
                 models=models,
                 client=client,
                 crawl_retries=crawl_retries,
                 skip_crawl=skip_crawl,
                 skip_clean=skip_clean,
                 skip_detect=skip_detect,
                 skip_headline_fix=skip_headline_fix,
                 skip_parse=skip_parse,
                 skip_annotate=skip_annotate,
                 skip_review=skip_review,
                 batch_detect=batch_detect,
                 batch_headline_fix=batch_headline_fix,
                 batch_annotate=batch_annotate,
                 batch_review=batch_review,
                 parallel_prompt=parallel_prompt)


if __name__ == '__main__':
    main()
