import os
import sys
import json
import time
import timeit
import datetime
import ollama
import logging
import argparse

from dotenv import load_dotenv
from openai import OpenAI
from typing import Callable, Union
from ollama import AsyncClient

from src import crawler, api_wrapper, util
from src.cleaner import Cleaner
from src.detector import Detector
from src.fixer import Fixer
from src.parser import Parser
from src.annotator import Annotator
from src.reviewer import Reviewer
from src.api_wrapper import ApiWrapper

DEFAULT_MODEL = "llama8b"


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


def run_batch(run_id: str, task: str, in_folder: str, func: Callable[[str, str], list[dict]]):
    # prepare and run the batch
    prepare_batch(run_id, task, in_folder, func)
    api_wrapper.run_batch(run_id, task)

    print(f"Batch {task} started. Waiting for completion...")

    # check the status of the batch every minute
    count = 0
    while True:
        count += 1
        batch_status = api_wrapper.check_batch_status(run_id, task)
        if batch_status.status == "completed":
            print(f"Batch completed after {count} checks, fetching results...")
            api_wrapper.get_batch_results(run_id, task)
            return True
        elif batch_status.status == "failed":
            print(f"Batch failed after {count} checks.")
            return False
        else:
            print(f"Batch still running after {count} checks...")
            time.sleep(60)


def run_pipeline(
        run_id: str,
        model: Union[str, dict],
        llm_api: ApiWrapper,
        id_file: str = None,
        single_pkg: str = None,
        crawl_retries: int = 2,
        skip_crawl: bool = False,
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
        parallel_processing: bool = False
):
    # prepare the output folders for the given run id
    if llm_api.client_type == "openai":
        util.prepare_output(run_id, model["name"], overwrite=False)
    else:
        util.prepare_output(run_id, model, overwrite=False)

    # crawl the privacy policies of the given packages if required
    if not skip_crawl:
        out_folder = f"output/{run_id}/original"
        if single_pkg is not None:
            crawler.crawl_list([single_pkg], out_folder, crawl_retries)
        else:
            crawler.execute(id_file, out_folder, crawl_retries)

    print(f"\nRunning pipeline for run id: {run_id} with model: {model}\n")
    logging.info(f"Running pipeline for run id: {run_id} with model: {model}")

    def process_package(pkg):
        print(f"Processing package: {pkg}")
        logging.info(f"Processing package: {pkg}")

        """ Pipeline elements """
        cleaner = Cleaner(run_id, pkg)
        detector = Detector(run_id, pkg, llm_api, model)
        fixer = Fixer(run_id, pkg, llm_api, model)
        parser = Parser(run_id, pkg)
        annotator = Annotator(run_id, pkg, llm_api, model)
        reviewer = Reviewer(run_id, pkg, llm_api, model)

        if not skip_clean:
            print(f"\rCleaning...{' ' * 10}", end='', flush=True)
            cleaner.execute()
        else:
            cleaner.skip()

        if not skip_detect:
            print(f"\rDetecting...{' ' * 10}", end='', flush=True)
            is_a_policy = detector.execute()
            if not is_a_policy:
                print(f"Package {pkg} is not a policy. Skipping further processing.\n")
                logging.warning(f"Package {pkg} is not a policy. Skipping further processing.")
                return
        else:
            detector.skip()

        if not skip_headline_fix:
            print(f"\rFixing headlines...{' ' * 10}", end='', flush=True)
            fixer.execute()
        else:
            fixer.skip()

        if not skip_parse:
            print(f"\rParsing...{' ' * 10}", end='', flush=True)
            parser.execute()
        else:
            parser.skip()

        if not skip_annotate:
            if parallel_processing:
                print(f"\rAnnotating in parallel...{' ' * 10}", end='', flush=True)
                annotator.execute_parallel()
            else:
                print(f"\rAnnotating...{' ' * 10}", end='', flush=True)
                annotator.execute()
        else:
            annotator.skip()

        if not skip_review:
            if parallel_processing:
                print(f"\rReviewing in parallel...{' ' * 10}", end='', flush=True)
                reviewer.execute_parallel()
            else:
                print(f"\rReviewing...{' ' * 10}", end='', flush=True)
                reviewer.execute()
        else:
            reviewer.skip()

        print(f"\nPackage {pkg} processed.\n")

    if single_pkg is not None:
        process_package(single_pkg)
    else:
        print("Processing all packages...")
        logging.info("Processing all packages...")
        if not os.path.exists(f"output/{run_id}/id_file.txt"):
            logging.error(f"ID file not found for run ID {run_id}.")
            print(f"ID file not found for run ID {run_id}.")
            sys.exit(1)
        id_file = f"output/{run_id}/id_file.txt"
        with open(id_file, 'r') as file:
            lines = file.readlines()
            total_packages = len(lines)
            for index, line in enumerate(lines):
                print(f"Processing package {index + 1} of {total_packages}")
                process_package(line.strip())


def main():
    logging.Formatter.converter = time.localtime
    logging.basicConfig(
        filename='open_source.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s'
    )
    logging.getLogger('httpx').setLevel(logging.DEBUG)

    ollama_models = [model_name for model_name in api_wrapper.OLLAMA_MODELS.keys()]
    # TODO: change to correct values for openai models
    openai_models = [model_name for model_name in api_wrapper.OPENAI_MODELS.keys()]

    parser = argparse.ArgumentParser(description="Run the pipeline with specified parameters.")
    parser.add_argument("-help", action="help", help="Show this help message and exit")
    parser.add_argument("-run-id", required=True, help="Run ID for the pipeline")
    parser.add_argument("-llm-service", required=True, choices=["openai", "ollama"], help="LLM service to use (openai or ollama)")
    parser.add_argument("-model", choices=ollama_models + openai_models,
                        help="Model to use for the pipeline")
    parser.add_argument("-id-file", action="store_true", help="Toggle use of an ID file for multiple packages to process")
    parser.add_argument("-pkg", help="Single package to process")
    parser.add_argument("-crawl-retries", type=int, default=2, help="Number of crawl retries")
    parser.add_argument("-no-crawl", action="store_true", help="Skip the crawl step")
    parser.add_argument("-no-clean", action="store_true", help="Skip the clean step")
    parser.add_argument("-no-detect", action="store_true", help="Skip the detect step")
    parser.add_argument("-no-headline-fix", action="store_true", help="Skip the headline fix step")
    parser.add_argument("-no-parse", action="store_true", help="Skip the parse step")
    parser.add_argument("-no-annotate", action="store_true", help="Skip the annotate step")
    parser.add_argument("-no-review", action="store_true", help="Skip the review step")
    parser.add_argument("-batch-detect", action="store_true", help="Run detect step in batch mode")
    parser.add_argument("-batch-headline-fix", action="store_true", help="Run headline fix step in batch mode")
    parser.add_argument("-batch-annotate", action="store_true", help="Run annotate step in batch mode")
    parser.add_argument("-batch-review", action="store_true", help="Run review step in batch mode")
    parser.add_argument("-model-file", action="store_true", help="Run pipeline for multiple models listed in models.txt")
    parser.add_argument("-parallel-off", action="store_false", help="Use parallel processing for local inference")
    parser.add_argument("-debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.pkg and args.id_file:
        parser.error("The -id-file option cannot be used with the -pkg option.")

    if not args.model and not args.model_file:
        parser.error("Either -model or -model-file must be provided.")

    if args.model and args.model_file:
        parser.error("The -model option cannot be used with the -model-file option.")

    if args.llm_service == "ollama" and (args.batch_detect or args.batch_headline_fix or args.batch_annotate or args.batch_review):
        parser.error("Batch processing is not supported for local inference with Ollama.")

    if args.llm_service == "openai":
        load_dotenv()

        if args.parallel_off:
            parser.error("Parallel processing is only supported for local inference with Ollama.")
        else:
            parallel_processing = False
    else:
        if not args.parallel_off:
            parallel_processing = False
        else:
            parallel_processing = True

    # initialize the correct client and model list based on the selected LLM service
    if args.llm_service == "openai":
        logging.info("OpenAI selected.")
        print("OpenAI selected.\n")
        llm_client = OpenAI()
        llm_names = openai_models
    elif args.llm_service == "ollama":
        logging.info("Ollama selected.")
        print("Ollama selected.\n")
        llm_client = AsyncClient()
        llm_names = [model_code for model_code in api_wrapper.OLLAMA_MODELS.keys()]
    else:
        logging.error(f"Invalid LLM service selected: {args.llm_service}")
        print("Invalid LLM service selected.\n")
        sys.exit(1)

    # initialize the api wrapper based on the selected LLM service
    llm_api = ApiWrapper(llm_client)

    # if the model file option is selected, read the models from the models.txt file
    if args.model_file:
        models_file_path = f"output/{args.run_id}/models.txt"
        if not os.path.exists(models_file_path):
            logging.error(f"Models file not found at {models_file_path}.")
            print(f"Models file not found at {models_file_path}.")
            sys.exit(1)
        file_content = util.read_from_file(models_file_path)
        models = [model_name.strip() for model_name in file_content.splitlines()]
        # remove empty lines
        models = list(filter(None, models))
        if args.llm_service == "ollama":
            models = [api_wrapper.OLLAMA_MODELS[model] for model in models]
    else:
        if args.llm_service == "ollama":
            models = [api_wrapper.OLLAMA_MODELS[args.model]]
        else:
            models = [args.model]

    if args.llm_service == "ollama":
        known_models = [api_wrapper.OLLAMA_MODELS[model_name] for model_name in llm_names]
    else:
        known_models = llm_names

    start_time_run = timeit.default_timer()

    for model_code in models:
        # check if the model is in the list of known models
        if model_code not in known_models:
            logging.error(f"Model {model_code} not in known model list.")
            logging.error(f"Known models: {known_models}")
            print(f"Model {model_code} not in known model list.")
            print(f"Please select one of the following models: {known_models}")
            continue
        else:
            # handle model downloading if the model is in the known list but not downloaded yet for local models
            if args.llm_service == "ollama":
                downloaded_models = llm_api.downloaded_models
                if model_code not in downloaded_models:
                    logging.debug(f"Downloaded models: {downloaded_models}")
                    logging.warning(f"Model {model_code} not in downloaded models. Downloading model {model_code}...")
                    print(f"Model {model_code} not in known model list. Downloading model...")
                    try:
                        ollama.pull(model_code)
                        logging.info(f"Model {model_code} downloaded.")
                        print(f"Model {model_code} downloaded.")
                    except Exception as e:
                        logging.error(f"Error downloading model {model_code}: {e}", exc_info=True)
                        print(f"Error downloading model {model_code}: {e}")
                        raise e
                else:
                    logging.info(f"Model {model_code} already downloaded.")
                    logging.info(f"Using model {model_code}.")

        logging.info(f"Using model {model_code} (model {models.index(model_code) + 1}/{len(models)}).")
        print(f"Using model {model_code} (model {models.index(model_code) + 1}/{len(models)}).\n")

        # pre-load the model if using local inference for faster processing
        if args.llm_service == "ollama":
            print(f"Loading model {model_code}...")
            llm_api.load_model(model_code)
            print(f"Model {model_code} loaded.\n")
            model = model_code
        else:
            model = api_wrapper.OPENAI_MODELS[model_code]

        print(f'Running pipeline using {"local inference" if args.llm_service == "ollama" else "OpenAI"} and the following parameters...')
        print(f'  skip_crawl: {args.no_crawl}')
        print(f'  skip_clean: {args.no_clean}')
        print(f'  skip_detect: {args.no_detect}')
        print(f'  skip_headline_fix: {args.no_headline_fix}')
        print(f'  skip_parse: {args.no_parse}')
        print(f'  skip_annotate: {args.no_annotate}')
        print(f'  skip_review: {args.no_review}')
        print(f'  batch_detect: {args.batch_detect}')
        print(f'  batch_headline_fix: {args.batch_headline_fix}')
        print(f'  batch_annotate: {args.batch_annotate}')
        print(f'  batch_review: {args.batch_review}')
        print(f'  parallel_processing: {parallel_processing}')

        skip_steps = any([
            args.no_crawl, args.no_clean, args.no_detect, args.no_headline_fix,
            args.no_parse, args.no_annotate, args.no_review,
            args.batch_detect, args.batch_headline_fix, args.batch_annotate, args.batch_review
        ])

        start_time_pipeline = timeit.default_timer()
        try:
            run_pipeline(
                run_id=args.run_id,
                model=model,
                llm_api=llm_api,
                id_file=args.id_file,
                single_pkg=args.pkg,
                crawl_retries=args.crawl_retries,
                skip_crawl=args.no_crawl if skip_steps else False,
                skip_clean=args.no_clean if skip_steps else False,
                skip_detect=args.no_detect if skip_steps else False,
                skip_headline_fix=args.no_headline_fix if skip_steps else False,
                skip_parse=args.no_parse if skip_steps else False,
                skip_annotate=args.no_annotate if skip_steps else False,
                skip_review=args.no_review if skip_steps else False,
                batch_detect=args.batch_detect if skip_steps else False,
                batch_headline_fix=args.batch_headline_fix if skip_steps else False,
                batch_annotate=args.batch_annotate if skip_steps else False,
                batch_review=args.batch_review if skip_steps else False,
                parallel_processing=parallel_processing,
            )
            end_time_pipeline = timeit.default_timer()
            logging.info(f"\n\nPipeline completed in {datetime.timedelta(seconds=end_time_pipeline - start_time_pipeline)}")
            print(f"\n\nPipeline completed in {datetime.timedelta(seconds=end_time_pipeline - start_time_pipeline)}")
        except Exception as e:
            logging.error(f"Error running pipeline: {e}", exc_info=True)
            print(f"Error running pipeline: {e}")
            end_time_pipeline = timeit.default_timer()
            logging.info(f"\n\nPipeline failed after {datetime.timedelta(seconds=end_time_pipeline - start_time_pipeline)}")
            print(f"\n\nPipeline failed after {datetime.timedelta(seconds=end_time_pipeline - start_time_pipeline)}")
            raise e

        llm_api.unload_model(model_code)

    end_time_run = timeit.default_timer()
    logging.info(f"\n\nRun \"{args.run_id}\" completed in {datetime.timedelta(seconds=end_time_run - start_time_run)}")
    print(f"\n\nRun \"{args.run_id}\" completed in {datetime.timedelta(seconds=end_time_run - start_time_run)}")


if __name__ == '__main__':
    main()

# com.aim.racing
# com.tripadvisor.tripadvisor

# 1 1. 1) (1) a a. a) (a) i i. i) (i) A A. A) (A) I I. I) (I)
