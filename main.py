import os
import sys
import json
import time
import timeit
import datetime

import ollama
import logging
import argparse
from typing import Callable
from ollama import AsyncClient

from src import crawler, api_wrapper, util
from src.cleaner import Cleaner
from src.detector import Detector
from src.fixer import Fixer
from src.parser import Parser
from src.annotator import Annotator

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
        model: str,
        ollama_client: AsyncClient,
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
        batch_review: bool = False):

    # prepare the output folders for the given run id
    util.prepare_output(run_id, model, overwrite=False)

    # crawl the privacy policies of the given packages if required
    if not skip_crawl:
        out_folder = f"output/{run_id}/original"
        if single_pkg is not None:
            crawler.crawl_list([single_pkg], out_folder, crawl_retries)
        else:
            crawler.execute(id_file, out_folder, crawl_retries)

    print(f"Running pipeline for run id: {run_id} with model: {model}")
    logging.info(f"Running pipeline for run id: {run_id} with model: {model}")

    def process_package(pkg):
        print(f"Processing package: {pkg}")
        logging.info(f"Processing package: {pkg}")

        cleaner = Cleaner(run_id, pkg)
        detector = Detector(run_id, pkg, ollama_client)
        fixer = Fixer(run_id, pkg, ollama_client)
        parser = Parser(run_id, pkg)
        annotator = Annotator(run_id, pkg, ollama_client)

        if not skip_clean:
            cleaner.execute()
        else:
            cleaner.skip()

        if not skip_detect:
            is_a_policy = detector.execute()
            if not is_a_policy:
                print(f"Package {pkg} is not a policy. Skipping further processing.\n")
                logging.warning(f"Package {pkg} is not a policy. Skipping further processing.")
                return
        else:
            detector.skip()

        if not skip_headline_fix:
            fixer.execute()
        else:
            fixer.skip()

        if not skip_parse:
            parser.execute()
        else:
            parser.skip()

        if not skip_annotate:
            if batch_annotate:
                annotator.execute_batched()
            else:
                annotator.execute()
        else:
            annotator.skip()

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
            for line in file:
                process_package(line.strip())


def main():
    logging.Formatter.converter = time.localtime
    logging.basicConfig(
        filename='open_source.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s'
    )
    logging.getLogger('httpx').setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Run the pipeline with specified parameters.")
    parser.add_argument("-help", action="help", help="Show this help message and exit")
    parser.add_argument("-run-id", required=True, help="Run ID for the pipeline")
    parser.add_argument("-model", default=DEFAULT_MODEL, choices=[model_name for model_name in api_wrapper.models.keys()],
                        help="Model to use for the pipeline")
    parser.add_argument("-id-file", action="store_true", help="Toggle use of an ID file for multiple packages to process")
    parser.add_argument("-pkg", help="Single package to process")
    parser.add_argument("-crawl-retries", type=int, default=2, help="Number of crawl retries")
    parser.add_argument("-debug", action="store_true", help="Enable debug logging")
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

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.pkg and args.id_file:
        parser.error("The -id-file option cannot be used with the -pkg option.")

    ollama_client = AsyncClient()

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
    else:
        models = [args.model]

    for model in models:
        if model not in api_wrapper.models.keys():
            logging.error(f"Model {model} not in known model list.")
            logging.error(f"Known models: {list(api_wrapper.models.keys())}")
            print(f"Model {model} not in known model list.")
            print(f"Please select one of the following models: {api_wrapper.models.keys()}")
            continue
        else:
            downloaded_models = [model['name'] for model in ollama.list()['models']]
            if api_wrapper.models[model] not in downloaded_models:
                logging.debug(f"Downloaded models: {downloaded_models}")
                logging.warning(f"Model {model} not in downloaded models. Downloading model {model}...")
                print(f"Model {model} not in known model list. Downloading model...")
                try:
                    ollama.pull(api_wrapper.models[model])
                    logging.info(f"Model {api_wrapper.models[model]} downloaded.")
                    print(f"Model {api_wrapper.models[model]} downloaded.")
                    os.environ['LLM_MODEL'] = model
                except Exception as e:
                    logging.error(f"Error downloading model {api_wrapper.models[model]}: {e}", exc_info=True)
                    print(f"Error downloading model {model}: {e}")
                    logging.warning(f"Proceeding with default model {DEFAULT_MODEL}.")
                    os.environ['LLM_MODEL'] = DEFAULT_MODEL
                    print(f"Proceeding with default model {DEFAULT_MODEL}.")
            else:
                logging.info(f"Model {model} already downloaded.")
                logging.info(f"Using model {model}.")
                os.environ['LLM_MODEL'] = model

        model_code = api_wrapper.models[os.environ.get('LLM_MODEL', DEFAULT_MODEL)]
        logging.info(f"Using model {model_code}.")
        print(f"Using model {model_code}.")

        api_wrapper.load_model(ollama_client, model_code)

        print(f'Running pipeline with the arguments: ')
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

        skip_steps = any([
            args.no_crawl, args.no_clean, args.no_detect, args.no_headline_fix,
            args.no_parse, args.no_annotate, args.no_review,
            args.batch_detect, args.batch_headline_fix, args.batch_annotate, args.batch_review
        ])

        start_time = timeit.default_timer()
        try:
            run_pipeline(
                run_id=args.run_id,
                model=model_code,
                ollama_client=ollama_client,
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
                batch_review=args.batch_review if skip_steps else False
            )
            end_time = timeit.default_timer()
            logging.info(f"\n\nPipeline completed in {datetime.timedelta(seconds=end_time - start_time)}")
            print(f"\n\nPipeline completed in {datetime.timedelta(seconds=end_time - start_time)}")
        except Exception as e:
            logging.error(f"Error running pipeline: {e}", exc_info=True)
            print(f"Error running pipeline: {e}")
            end_time = timeit.default_timer()
            logging.info(f"\n\nPipeline failed after {datetime.timedelta(seconds=end_time - start_time)}")
            print(f"\n\nPipeline failed after {datetime.timedelta(seconds=end_time - start_time)}")
            raise e

        api_wrapper.unload_model(ollama_client, model_code)


if __name__ == '__main__':
    main()

# com.aim.racing
# com.tripadvisor.tripadvisor

# 1 1. 1) (1) a a. a) (a) i i. i) (i) A A. A) (A) I I. I) (I)
