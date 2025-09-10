"""
Main entry point for the policy analysis pipeline.
This script handles both command-line and WebSocket-based execution of the pipeline.
"""

import asyncio
import json
import sys
from typing import Dict, Any

import websockets
from dotenv import load_dotenv

from src.pipeline_executor import PipelineExecutor, PipelineStatus
from src.state_manager import WebSocketStateManager, ConsoleStateManager, BaseStateManager


async def handle_websocket(websocket):
    """Handle WebSocket connections and pipeline control messages."""

    print(f"\n[WebSocket] New connection established")
    state_manager = WebSocketStateManager()
    state_manager.connections.add(websocket)
    try:
        await state_manager.broadcast_state()
        async for message in websocket:
            print(f"\n[WebSocket] Received message: {message}")
            data = json.loads(message)
            message_type = data.get("type")

            if message_type is None:
                continue

            elif message_type == "start_pipeline":
                if state_manager.state.status == PipelineStatus.RUNNING:
                    await state_manager.raise_error("Cannot start pipeline while another is running")
                    continue

                config = data.get("config")
                if config is None:
                    await state_manager.raise_error(error_message="No config provided", abort=True)
                    continue

                else:
                    print(f"[WebSocket] Starting pipeline with config: {json.dumps(config, indent=2)}")
                    run_id, pkg, default_model, models, crawl_retries, skip_crawl, skip_clean, skip_detect, skip_parse, skip_annotate, skip_review, skip_classify, skip_targeted_annotate, batch_detect, batch_annotate, batch_review, batch_classify, batch_targeted_annotate, parallel_prompt, hostname, use_two_step_annotation, confidence_threshold, use_rag, max_examples_per_requirement, min_example_confidence, annotated_folder, use_opp_115, root_dir = parse_config(config)

                    # Create and run the executor directly in the event loop
                    executor = PipelineExecutor(
                        run_id=run_id,
                        pkg=pkg,
                        default_model=default_model,
                        models=models,
                        crawl_retries=crawl_retries,
                        skip_crawl=skip_crawl,
                        skip_clean=skip_clean,
                        skip_detect=skip_detect,
                        skip_parse=skip_parse,
                        skip_annotate=skip_annotate,
                        skip_review=skip_review,
                        skip_classify=skip_classify,
                        skip_targeted_annotate=skip_targeted_annotate,
                        batch_detect=batch_detect,
                        batch_annotate=batch_annotate,
                        batch_review=batch_review,
                        batch_classify=batch_classify,
                        batch_targeted_annotate=batch_targeted_annotate,
                        parallel_prompt=parallel_prompt,
                        hostname=hostname,
                        state_manager=state_manager,
                        use_two_step_annotation=use_two_step_annotation,
                        confidence_threshold=confidence_threshold,
                        use_rag=use_rag,
                        max_examples_per_requirement=max_examples_per_requirement,
                        min_example_confidence=min_example_confidence,
                        annotated_folder=annotated_folder,
                        use_opp_115=use_opp_115,
                        root_dir=root_dir
                    )
                    await executor.execute()

            elif message_type == "stop_pipeline":
                print("[WebSocket] Stopping pipeline")
                await state_manager.update_state(
                    PipelineStatus.ABORTED,
                    message="Stopping pipeline..."
                )

    except Exception as e:
        await state_manager.raise_error(error_message=str(e), abort=True)
    finally:
        try:
            state_manager.connections.remove(websocket)
            await state_manager.update_state(PipelineStatus.COMPLETED)
            print("[WebSocket] Connection closed")
        except KeyError:
            pass


async def start_websocket_server(host: str = "localhost", port: int = 8765):
    """Start the WebSocket server for real-time pipeline updates."""
    async with websockets.serve(handle_websocket, host, port):
        print(f"WebSocket server started on ws://{host}:{port}")
        await asyncio.Future()  # run forever


def parse_args():
    # parse the run id from the command line argument "-run-id"
    run_id = None
    if "-run-id" in sys.argv:
        idx = sys.argv.index("-run-id")
        if idx + 1 < len(sys.argv):
            run_id = sys.argv[idx + 1]

    # parse the pkg from the command line argument "-pkg"
    pkg = None
    if "-pkg" in sys.argv:
        idx = sys.argv.index("-pkg")
        if idx + 1 < len(sys.argv):
            pkg = sys.argv[idx + 1]

    # parse the model from the command line argument "-model"
    default_model = None
    if "-model" in sys.argv:
        idx = sys.argv.index("-model")
        if idx + 1 < len(sys.argv):
            default_model = sys.argv[idx + 1]

    # parse the step-specific models from the command line arguments "-model-<step>"
    models = {
        "detect": None,
        "annotate": None,
        "review": None,
        "targeted_annotate": None
    }
    for step in models.keys():
        if f"-model-{step}" in sys.argv:
            idx = sys.argv.index(f"-model-{step}")
            if idx + 1 < len(sys.argv):
                models[step] = sys.argv[idx + 1]

    # parse the number of crawl retries from the command line argument "-crawl-retries"
    crawl_retries = 2
    if "-crawl-retries" in sys.argv:
        idx = sys.argv.index("-crawl-retries")
        if idx + 1 < len(sys.argv):
            crawl_retries = int(sys.argv[idx + 1])

    # parse command line args that indicate which steps to skip
    skip_crawl = "-skip-crawl" in sys.argv
    skip_clean = "-skip-clean" in sys.argv
    skip_detect = "-skip-detect" in sys.argv
    skip_parse = "-skip-parse" in sys.argv
    skip_annotate = "-skip-annotate" in sys.argv
    skip_review = "-skip-review" in sys.argv
    skip_classify = "-skip-classify" in sys.argv
    skip_targeted_annotate = "-skip-targeted-annotate" in sys.argv

    # parse command line args that indicate which LLM-reliant steps to run in batch mode (if supported by the client)
    batch_detect = "-batch-detect" in sys.argv
    batch_annotate = "-batch-annotate" in sys.argv
    batch_review = "-batch-review" in sys.argv
    batch_classify = "-batch-classify" in sys.argv
    batch_targeted_annotate = "-batch-targeted-annotate" in sys.argv

    # parse whether to use parallel prompts from the command line argument "-parallel-prompt"
    parallel_prompt = "-parallel-prompt" in sys.argv

    # parse whether to use two-step annotation pipeline
    use_two_step_annotation = "-two-step-annotation" in sys.argv

    # Warn if trying to skip classify/targeted_annotate without two-step annotation
    if skip_classify and not use_two_step_annotation:
        print("Warning: -skip-classify only has an effect when using -two-step-annotation")
    if skip_targeted_annotate and not use_two_step_annotation:
        print("Warning: -skip-targeted-annotate only has an effect when using -two-step-annotation")

    # parse annotated folder for cheating classifier (evaluation mode)
    annotated_folder = None
    if "-annotated-folder" in sys.argv:
        idx = sys.argv.index("-annotated-folder")
        if idx + 1 < len(sys.argv):
            annotated_folder = sys.argv[idx + 1]

    # parse confidence threshold for targeted annotation
    confidence_threshold = 0.3
    if "-confidence-threshold" in sys.argv:
        idx = sys.argv.index("-confidence-threshold")
        if idx + 1 < len(sys.argv):
            confidence_threshold = float(sys.argv[idx + 1])

    # parse RAG configuration
    use_rag = "-use-rag" in sys.argv
    no_rag = "-no-rag" in sys.argv
    if no_rag:
        use_rag = False
    
    max_examples_per_requirement = 2
    if "-max-examples" in sys.argv:
        idx = sys.argv.index("-max-examples")
        if idx + 1 < len(sys.argv):
            max_examples_per_requirement = int(sys.argv[idx + 1])
    
    min_example_confidence = 0.8
    if "-min-example-confidence" in sys.argv:
        idx = sys.argv.index("-min-example-confidence")
        if idx + 1 < len(sys.argv):
            min_example_confidence = float(sys.argv[idx + 1])

    # parse OPP-115 scheme configuration
    use_opp_115 = "-opp-115" in sys.argv

    # parse the hostname from the command line argument "-hostname"
    hostname = None
    if "-hostname" in sys.argv:
        idx = sys.argv.index("-hostname")
        if idx + 1 < len(sys.argv):
            hostname = sys.argv[idx + 1]

    # parse the root directory from the command line argument "-root-dir"
    root_dir = None
    if "-root-dir" in sys.argv:
        idx = sys.argv.index("-root-dir")
        if idx + 1 < len(sys.argv):
            root_dir = sys.argv[idx + 1]

    return (
        run_id,
        pkg,
        default_model,
        models,
        crawl_retries,
        skip_crawl,
        skip_clean,
        skip_detect,
        skip_parse,
        skip_annotate,
        skip_review,
        skip_classify,
        skip_targeted_annotate,
        batch_detect,
        batch_annotate,
        batch_review,
        batch_classify,
        batch_targeted_annotate,
        parallel_prompt,
        hostname,
        use_two_step_annotation,
        confidence_threshold,
        use_rag,
        max_examples_per_requirement,
        min_example_confidence,
        annotated_folder,
        use_opp_115,
        root_dir
    )


def parse_config(config: dict):
    return (
        config.get("run_id", None),
        config.get("pkg", None),
        config.get("default_model", None),
        config.get("models", None),
        config.get("crawl_retries", 2),
        config.get("skip_crawl", False),
        config.get("skip_clean", False),
        config.get("skip_detect", False),
        config.get("skip_parse", False),
        config.get("skip_annotate", False),
        config.get("skip_review", False),
        config.get("skip_classify", False),
        config.get("skip_targeted_annotate", False),
        config.get("batch_detect", False),
        config.get("batch_annotate", False),
        config.get("batch_review", False),
        config.get("batch_classify", False),
        config.get("batch_targeted_annotate", False),
        config.get("parallel_prompt", False),
        config.get("hostname", None),
        config.get("use_two_step_annotation", False),
        config.get("confidence_threshold", 0.3),
        config.get("use_rag", True),
        config.get("max_examples_per_requirement", 2),
        config.get("min_example_confidence", 0.8),
        config.get("annotated_folder", None),
        config.get("use_opp_115", False),
        config.get("root_dir", None)
    )


def prepare_pipeline(
        run_id: str,
        pkg: str,
        default_model: str,
        models: dict = None,
        crawl_retries: int = 2,
        skip_crawl: bool = False,
        skip_clean: bool = False,
        skip_detect: bool = False,
        skip_parse: bool = False,
        skip_annotate: bool = False,
        skip_review: bool = False,
        skip_classify: bool = False,
        skip_targeted_annotate: bool = False,
        batch_detect: bool = False,
        batch_annotate: bool = False,
        batch_review: bool = False,
        batch_classify: bool = False,
        batch_targeted_annotate: bool = False,
        parallel_prompt: bool = False,
        hostname: str = None,
        use_two_step_annotation: bool = False,
        confidence_threshold: float = 0.3,
        use_rag: bool = True,
        max_examples_per_requirement: int = 2,
        min_example_confidence: float = 0.8,
        annotated_folder: str = None,
        use_opp_115: bool = False,
        root_dir: str = None,
        state_manager: BaseStateManager = None
):
    # create a state manager for the pipeline if none is provided
    if state_manager is None:
        state_manager = ConsoleStateManager()
        spawn_event_loop = True
    else:
        spawn_event_loop = False

        state_manager.update_state(
            status=PipelineStatus.INITIALIZING,
            current_step="Initializing",
            message="Starting pipeline..."
        )

    # Create the executor
    executor = PipelineExecutor(
        run_id,
        pkg,
        default_model,
        models,
        crawl_retries,
        skip_crawl,
        skip_clean,
        skip_detect,
        skip_parse,
        skip_annotate,
        skip_review,
        skip_classify,
        skip_targeted_annotate,
        batch_detect,
        batch_annotate,
        batch_review,
        batch_classify,
        batch_targeted_annotate,
        parallel_prompt,
        hostname,
        state_manager,
        use_two_step_annotation,
        confidence_threshold,
        use_rag,
        max_examples_per_requirement,
        min_example_confidence,
        annotated_folder,
        use_opp_115,
        root_dir
    )

    # run the pipeline in an event loop, spawning a new one if not already in an async context
    if spawn_event_loop:
        asyncio.run(executor.execute())
    else:
        return executor.execute()  # Return the coroutine to be awaited by the caller


def main():
    """Main entry point for the application."""

    # load environment variables from the .env file
    load_dotenv()

    # Check for help flag or no arguments
    if "--help" in sys.argv or "-h" in sys.argv or len(sys.argv) == 1:
        print("""
Policy Analysis Pipeline

USAGE:
    python main.py [OPTIONS]

MODES:
    --websocket                 Start WebSocket server for real-time pipeline control
    --config <path>             Load configuration from JSON file

REQUIRED ARGUMENTS:
    -run-id <id>                Unique identifier for this pipeline run

MODEL CONFIGURATION:
    -model <model>              Default model to use for all LLM steps
    -model-detect <model>       Specific model for policy detection step
    -model-annotate <model>     Specific model for annotation step
    -model-review <model>       Specific model for review step
    -model-targeted-annotate <model> Specific model for targeted annotation step

PIPELINE CONTROL:
    -skip-crawl                 Skip the crawling steps (metadata and policy crawling)
    -skip-clean                 Skip the cleaning step
    -skip-detect                Skip the policy detection step
    -skip-parse                 Skip the parsing step
    -skip-annotate              Skip the annotation step
    -skip-review                Skip the review step
    -skip-classify              Skip the classification step (only with -two-step-annotation)
    -skip-targeted-annotate     Skip the targeted annotation step (only with -two-step-annotation)

TWO-STEP ANNOTATION PIPELINE:
    -two-step-annotation        Use two-step annotation pipeline (classify + targeted annotate)
    -annotated-folder <path>    Path to folder with manually annotated policies (evaluation mode)
    -confidence-threshold <float> Confidence threshold for targeted annotation (default: 0.3)

RAG CONFIGURATION:
    -use-rag                    Enable RAG (Retrieval-Augmented Generation) for examples (default: enabled)
    -no-rag                     Disable RAG for examples
    -max-examples <int>         Maximum examples per requirement (default: 2)
    -min-example-confidence <float> Minimum confidence for examples (default: 0.8)

ANNOTATION SCHEME:
    -opp-115                    Use OPP-115 privacy practice annotation scheme (default: GDPR transparency requirements)

BATCH PROCESSING:
    -batch-detect               Run detection step in batch mode
    -batch-annotate             Run annotation step in batch mode
    -batch-review               Run review step in batch mode
    -batch-classify             Run classification step in batch mode (two-step pipeline)
    -batch-targeted-annotate    Run targeted annotation step in batch mode (two-step pipeline)

PERFORMANCE OPTIONS:
    -parallel-prompt            Enable parallel prompt processing
    -crawl-retries <num>        Number of crawl retries (default: 2)
    -hostname <hostname>        Specify hostname for processing
    -root-dir <path>            Specify root directory for output (default: ../../output)

HELP:
    --help, -h                  Show this help message

EXAMPLES:
    # Basic pipeline run
    python main.py -run-id my-analysis -model gpt-4

    # Two-step annotation pipeline
    python main.py -run-id my-analysis -model gpt-4 -two-step-annotation

    # Two-step pipeline with evaluation mode (cheating classifier)
    python main.py -run-id my-analysis -model gpt-4 -two-step-annotation -annotated-folder /path/to/annotated/policies

    # Two-step pipeline with custom confidence threshold
    python main.py -run-id my-analysis -model gpt-4 -two-step-annotation -confidence-threshold 0.5

    # Two-step pipeline with RAG disabled
    python main.py -run-id my-analysis -model gpt-4 -two-step-annotation -no-rag

    # Two-step pipeline with custom RAG settings
    python main.py -run-id my-analysis -model gpt-4 -two-step-annotation -max-examples 3 -min-example-confidence 0.9

    # Skip certain steps
    python main.py -run-id my-analysis -model gpt-4 -skip-clean -skip-parse

    # Skip classification step (only works with two-step annotation)
    python main.py -run-id my-analysis -model gpt-4 -two-step-annotation -skip-classify

    # Use different models for different steps
    python main.py -run-id my-analysis -model gpt-4 -model-detect claude-3

    # Run in batch mode with parallel processing
    python main.py -run-id my-analysis -model gpt-4 -batch-detect -batch-annotate -parallel-prompt

    # Start WebSocket server
    python main.py --websocket

    # Load from configuration file
    python main.py --config config.json

    # Specify custom output directory
    python main.py -run-id my-analysis -model gpt-4 -root-dir /path/to/output
        """)
        return

    # check if we should start the websocket server, in which case the pipeline will be triggered by the websocket messages
    if "--websocket" in sys.argv:
        asyncio.run(start_websocket_server())
    else:
        # check if we should load a configuration file
        config_path = None
        if "--config" in sys.argv:
            idx = sys.argv.index("--config")
            if idx + 1 < len(sys.argv):
                config_path = sys.argv[idx + 1]

        # parse the command line arguments or the configuration file if provided
        if config_path is not None:
            with open(config_path, 'r') as file:
                run_id, pkg, default_model, models, crawl_retries, skip_crawl, skip_clean, skip_detect, \
                    skip_parse, skip_annotate, skip_review, skip_classify, skip_targeted_annotate, \
                    batch_detect, batch_annotate, batch_review, batch_classify, batch_targeted_annotate, \
                    parallel_prompt, hostname, use_two_step_annotation, confidence_threshold, use_rag, max_examples_per_requirement, min_example_confidence, annotated_folder, use_opp_115, root_dir = parse_config(json.load(file))
        else:
            run_id, pkg, default_model, models, crawl_retries, skip_crawl, skip_clean, skip_detect, \
                skip_parse, skip_annotate, skip_review, skip_classify, skip_targeted_annotate, \
                batch_detect, batch_annotate, batch_review, batch_classify, batch_targeted_annotate, \
                parallel_prompt, hostname, use_two_step_annotation, confidence_threshold, use_rag, max_examples_per_requirement, min_example_confidence, annotated_folder, use_opp_115, root_dir = parse_args()

        # trigger the pipeline execution
        prepare_pipeline(
            run_id,
            pkg,
            default_model,
            models,
            crawl_retries,
            skip_crawl,
            skip_clean,
            skip_detect,
            skip_parse,
            skip_annotate,
            skip_review,
            skip_classify,
            skip_targeted_annotate,
            batch_detect,
            batch_annotate,
            batch_review,
            batch_classify,
            batch_targeted_annotate,
            parallel_prompt,
            hostname,
            use_two_step_annotation,
            confidence_threshold,
            use_rag,
            max_examples_per_requirement,
            min_example_confidence,
            annotated_folder,
            use_opp_115,
            root_dir
        )


if __name__ == '__main__':
    main()
