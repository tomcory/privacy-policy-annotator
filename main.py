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
                    run_id, default_model, models, skip_clean, skip_detect, skip_parse, skip_annotate, skip_review, batch_detect, batch_annotate, batch_review, parallel_prompt, hostname = parse_config(config)

                    # Create and run the executor directly in the event loop
                    executor = PipelineExecutor(
                        run_id=run_id,
                        default_model=default_model,
                        models=models,
                        skip_clean=skip_clean,
                        skip_detect=skip_detect,
                        skip_parse=skip_parse,
                        skip_annotate=skip_annotate,
                        skip_review=skip_review,
                        batch_detect=batch_detect,
                        batch_annotate=batch_annotate,
                        batch_review=batch_review,
                        parallel_prompt=parallel_prompt,
                        hostname=hostname,
                        state_manager=state_manager
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
    # parse the run id from the command line argument "-run-id" or exit if not provided
    run_id = None
    if "-run-id" in sys.argv:
        idx = sys.argv.index("-run-id")
        if idx + 1 < len(sys.argv):
            run_id = sys.argv[idx + 1]

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
        "review": None
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
    skip_clean = "-no-clean" in sys.argv
    skip_detect = "-no-detect" in sys.argv
    skip_parse = "-no-parse" in sys.argv
    skip_annotate = "-no-annotate" in sys.argv
    skip_review = "-no-review" in sys.argv

    # parse command line args that indicate which LLM-reliant steps to run in batch mode (if supported by the client)
    batch_detect = "-batch-detect" in sys.argv
    batch_annotate = "-batch-annotate" in sys.argv
    batch_review = "-batch-review" in sys.argv

    # parse whether to use parallel prompts from the command line argument "-parallel-prompt"
    parallel_prompt = "-parallel-prompt" in sys.argv

    # parse the hostname from the command line argument "-hostname"
    hostname = None
    if "-hostname" in sys.argv:
        idx = sys.argv.index("-hostname")
        if idx + 1 < len(sys.argv):
            hostname = sys.argv[idx + 1]

    return (
        run_id,
        default_model,
        models,
        skip_clean,
        skip_detect,
        skip_parse,
        skip_annotate,
        skip_review,
        batch_detect,
        batch_annotate,
        batch_review,
        parallel_prompt,
        hostname
    )


def parse_config(config: dict):
    return (
        config.get("run_id", None),
        config.get("default_model", None),
        config.get("models", None),
        config.get("skip_clean", False),
        config.get("skip_detect", False),
        config.get("skip_parse", False),
        config.get("skip_annotate", False),
        config.get("skip_review", False),
        config.get("batch_detect", False),
        config.get("batch_annotate", False),
        config.get("batch_review", False),
        config.get("parallel_prompt", False),
        config.get("hostname", None)
    )


def prepare_pipeline(
        run_id: str,
        default_model: str,
        models: dict = None,
        skip_clean: bool = False,
        skip_detect: bool = False,
        skip_parse: bool = False,
        skip_annotate: bool = False,
        skip_review: bool = False,
        batch_detect: bool = False,
        batch_annotate: bool = False,
        batch_review: bool = False,
        parallel_prompt: bool = False,
        hostname: str = None,
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
        default_model,
        models,
        skip_clean,
        skip_detect,
        skip_parse,
        skip_annotate,
        skip_review,
        batch_detect,
        batch_annotate,
        batch_review,
        parallel_prompt,
        hostname,
        state_manager
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

PIPELINE CONTROL:
    -no-clean                   Skip the cleaning step
    -no-detect                  Skip the policy detection step
    -no-parse                   Skip the parsing step
    -no-annotate                Skip the annotation step
    -no-review                  Skip the review step

BATCH PROCESSING:
    -batch-detect               Run detection step in batch mode
    -batch-annotate             Run annotation step in batch mode
    -batch-review               Run review step in batch mode

PERFORMANCE OPTIONS:
    -parallel-prompt            Enable parallel prompt processing
    -crawl-retries <num>        Number of crawl retries (default: 2)
    -hostname <hostname>        Specify hostname for processing

HELP:
    --help, -h                  Show this help message

EXAMPLES:
    # Basic pipeline run
    python main.py -run-id my-analysis -model gpt-4

    # Skip certain steps
    python main.py -run-id my-analysis -model gpt-4 -no-clean -no-parse

    # Use different models for different steps
    python main.py -run-id my-analysis -model gpt-4 -model-detect claude-3

    # Run in batch mode with parallel processing
    python main.py -run-id my-analysis -model gpt-4 -batch-detect -batch-annotate -parallel-prompt

    # Start WebSocket server
    python main.py --websocket

    # Load from configuration file
    python main.py --config config.json
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
                run_id, default_model, models, skip_clean, skip_detect, \
                    skip_parse, skip_annotate, skip_review, batch_detect, batch_annotate, \
                    batch_review, parallel_prompt, hostname = parse_config(json.load(file))
        else:
            run_id, default_model, models, skip_clean, skip_detect, \
                skip_parse, skip_annotate, skip_review, batch_detect, batch_annotate, \
                batch_review, parallel_prompt, hostname = parse_args()

        # trigger the pipeline execution
        prepare_pipeline(
            run_id,
            default_model,
            models,
            skip_clean,
            skip_detect,
            skip_parse,
            skip_annotate,
            skip_review,
            batch_detect,
            batch_annotate,
            batch_review,
            parallel_prompt,
            hostname
        )


if __name__ == '__main__':
    main()
