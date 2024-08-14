import json
import os
import sys
import time
from typing import Callable

from src import crawler, cleaner, detector, fixer, parser, annotator, reviewer, api_wrapper, util


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
        id_file: str = None,
        single_pkg: str = None,
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
        batch_review: bool = False):

    # define the processing steps of the pipeline
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

        if not skip:
            if single_pkg is not None:
                if is_batch_step:
                    run_batch(run_id, step, in_folder, func)
                else:
                    func(run_id, single_pkg, in_folder, out_folder, use_batch_result)
            else:
                if is_batch_step:
                    run_batch(run_id, step, in_folder, func)
                else:
                    for filename in os.listdir(in_folder):
                        # Ensure we are processing files only (not directories)
                        if os.path.isfile(os.path.join(in_folder, filename)):
                            pkg = os.path.splitext(filename)[0]
                            try:
                                func(run_id, pkg, in_folder, out_folder, use_batch_result)
                            except Exception as e:
                                with open(f"output/{run_id}/log/error.txt", 'w') as error_file:
                                    error_file.write(f"{pkg}\n")
        else:
            # copy the content of in_folder to out_folder without processing
            for filename in os.listdir(in_folder):
                if os.path.isfile(os.path.join(in_folder, filename)):
                    with open(in_folder + '/' + filename, 'r') as file:
                        content = file.read()
                        with open(out_folder + '/' + filename, 'w') as new_file:
                            new_file.write(content)


def main():

    # parse the run id from the command line argument "-run-id" or exit if not provided
    if "-run-id" in sys.argv:
        idx = sys.argv.index("-run-id")
        if idx + 1 < len(sys.argv):
            run_id = sys.argv[idx + 1]
        else:
            print("Error: No run id provided. Please provide a run id with the -run-id argument.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: No run id provided. Please provide a run id with the -run-id argument.", file=sys.stderr)
        sys.exit(1)

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

    # parse command line args that indicate which LLM-reliant steps to run in batch mode
    batch_detect = "-batch-detect" in sys.argv
    batch_headline_fix = "-batch-headline-fix" in sys.argv
    batch_annotate = "-batch-annotate" in sys.argv
    batch_review = "-batch-review" in sys.argv

    # if no "skip-*" or "batch-*" arguments are provided, call the pipeline with the default arguments
    if (not skip_crawl
            and not skip_clean
            and not skip_detect
            and not skip_headline_fix
            and not skip_parse
            and not skip_annotate
            and not skip_review
            and not batch_detect
            and not batch_headline_fix
            and not batch_annotate
            and not batch_review):
        run_pipeline(run_id=run_id,
                     id_file=id_file,
                     single_pkg=single_pkg,
                     crawl_retries=crawl_retries)
    else:
        run_pipeline(run_id=run_id,
                     id_file=id_file,
                     single_pkg=single_pkg,
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
                     batch_review=batch_review)


if __name__ == '__main__':
    main()

# com.aim.racing
# com.tripadvisor.tripadvisor

# 1 1. 1) (1) a a. a) (a) i i. i) (i) A A. A) (A) I I. I) (I)
