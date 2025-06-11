import asyncio
import json
import sys
import time
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm


class PipelineStatus(Enum):
    IDLE = "idle"  # Pipeline is not running
    INITIALIZING = "initializing"
    RUNNING = "running"  # Pipeline is currently processing
    COMPLETED = "completed"  # Pipeline has finished successfully
    INTERRUPTED = "interrupted" # Pipeline was interrupted by the user
    ERROR = "error"  # Pipeline encountered an error
    ABORTED = "aborted"  # Pipeline was aborted due to an error


@dataclass
class PipelineState:
    status: PipelineStatus =  PipelineStatus.IDLE  # Current pipeline status
    current_step: str = ""  # Name of the current processing step
    progress: float = 0.0  # Overall progress (0.0 to 1.0)
    step_progress: float = 0.0  # Progress within current step (0.0 to 1.0)
    file_progress: float = 0.0  # Progress within current file (0.0 to 1.0)
    current_file: str = ""  # Name of the file being processed
    total_files: int = 0  # Total number of files to process
    step_details: str = ""  # Additional details about the current step
    message: str = ""  # Status message
    warning: str = ""  # Warning message if any
    error: str = ""  # Error message if any


class BaseStateManager:
    """Abstract base class for state managers that handle pipeline state updates."""

    def __init__(self):
        self.state = PipelineState()
        self.previous_state = self.state
        self.last_update = time.time()

    async def raise_error(self, error_message: str, abort=False):
        """Raise an error and update the pipeline state."""
        if abort:
            await self.update_state(status=PipelineStatus.ABORTED, error=error_message)
            await self._abort()
        else:
            await self.update_state(status=PipelineStatus.ERROR, error=error_message)

    async def raise_warning(self, warning_message: str):
        """Raise a warning and update the pipeline state."""
        await self.update_state(warning=warning_message)

    async def update_state(
            self,
            status: PipelineStatus = None,
            current_step: str = None,
            progress: float = None,
            step_progress: float = None,
            file_progress: float = None,
            current_file: str = None,
            total_files: int = None,
            step_details: str = None,
            message: str = "",
            warning: str = "",
            error: str = ""
    ):
        """Internal coroutine to asynchronously update the pipeline state, called by self.update_state."""
        if status is None:
            status = self.state.status
        if current_step is None:
            current_step = self.state.current_step
        if progress is None:
            progress = self.state.progress
        if step_progress is None:
            step_progress = self.state.step_progress
        if file_progress is None:
            file_progress = self.state.file_progress
        if current_file is None:
            current_file = self.state.current_file
        if total_files is None:
            total_files = self.state.total_files
        if step_details is None:
            step_details = self.state.step_details

        time_since_last_update = time.time() - self.last_update

        # check whether the state has any changes
        if time_since_last_update < 1 and \
                status == self.state.status and \
                current_step == self.state.current_step and \
                progress == self.state.progress and \
                step_progress == self.state.step_progress and \
                file_progress == self.state.file_progress and \
                current_file == self.state.current_file and \
                total_files == self.state.total_files and \
                step_details == self.state.step_details and \
                message == self.state.message and \
                warning == self.state.warning and \
                error == self.state.error:
            return

        self.last_update = time.time()
        self.previous_state = self.state
        self.state = PipelineState(
            status=status,
            current_step=current_step,
            progress=progress,
            step_progress=step_progress,
            file_progress=file_progress,
            current_file=current_file,
            total_files=total_files,
            step_details=step_details,
            message=message,
            warning=warning,
            error=error
        )
        await self.broadcast_state()

    async def broadcast_state(self):
        """Broadcast the current state."""
        raise NotImplementedError("Subclasses must implement this method.")

    async def _abort(self):
        """Abort the pipeline."""
        raise NotImplementedError("Subclasses must implement this method.")


class WebSocketStateManager(BaseStateManager):
    """State manager that broadcasts state updates to connected WebSocket clients."""

    def __init__(self):
        super().__init__()
        self.connections = set()  # Set of connected WebSocket clients
        self.last_broadcast_time = 0  # Timestamp of the last broadcast
        self.minor_update_interval = 1.0  # Minimum time between minor updates in seconds

    async def broadcast_state(self):
        """Broadcast the current state, with rate limiting for minor updates."""

        # Skip broadcasting if there are no connections
        if not self.connections:
            return


        current_time = asyncio.get_event_loop().time()
        time_since_last_broadcast = current_time - self.last_broadcast_time

        # Check if it's a major update or enough time has passed to broadcast
        do_broadcast = (time_since_last_broadcast >= self.minor_update_interval or (
                self.previous_state.status != self.state.status or
                self.previous_state.current_step != self.state.current_step or
                self.previous_state.progress != self.state.progress or
                self.previous_state.current_file != self.state.current_file or
                self.previous_state.total_files != self.state.total_files or
                self.previous_state.step_details != self.state.step_details or
                self.previous_state.message != self.state.message or
                self.previous_state.warning != self.state.warning or
                self.previous_state.error != self.state.error))

        message = json.dumps({
            "type": "state_update",
            "state": {
                "status": self.state.status.value,
                "current_step": self.state.current_step,
                "progress": self.state.progress,
                "step_progress": self.state.step_progress,
                "file_progress": self.state.file_progress,
                "current_file": self.state.current_file,
                "total_files": self.state.total_files,
                "step_details": self.state.step_details,
                "message": self.state.message,
                "error": self.state.error
            }
        })

        # Broadcast the state if there are changes or enough time has passed
        if do_broadcast:
            print(f"\n[State] Broadcasting: {json.dumps(json.loads(message), indent=2)}")
            await asyncio.gather(*[conn.send(message) for conn in self.connections])
            self.last_broadcast_time = current_time
        else:
            print(f"\n[State] Skipping broadcast of minor state update due to rate limiting: {json.dumps(json.loads(message), indent=2)}")

    async def _abort(self):
        """Abort the pipeline and notify all connected clients."""
        for conn in self.connections:
            await conn.close()
        self.connections.clear()
        print("\nPipeline aborted")
        sys.exit(1)


class ConsoleStateManager(BaseStateManager):
    """State manager that displays progress in the console using tqdm."""

    def __init__(self):
        super().__init__()
        self.pbar = None  # Progress bar instance
        self.last_step = None  # Track last step for step change detection
        self.last_file = None  # Track last file for file change detection

    async def broadcast_state(self):
        """Display the current state in the console with appropriate formatting."""

        # Update or create progress bar for running steps with multiple files
        if (self.state.status == PipelineStatus.RUNNING and
                self.state.total_files >= 1):

            if self.pbar is None or self.pbar.total != self.state.total_files:
                # Close existing bar if total files changed
                if self.pbar is not None:
                    self.pbar.close()

                # Create new progress bar
                self.pbar = tqdm(
                    total=self.state.total_files,
                    desc=f"{self.state.current_step}: {self.state.current_file} ({self.state.file_progress:.2%})",
                    unit="file",
                    ncols=160,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )

            # Update progress bar position
            current_position = int(self.state.step_progress * self.state.total_files)
            self.pbar.n = current_position
            if self.state.current_file:
                self.pbar.set_description(
                    f"{self.state.current_step}: {self.state.current_file} ({self.state.file_progress:.2%})")
            self.pbar.refresh()
        
        # Handle status changes
        if self.state.status != self.previous_state.status:
            if self.state.status == PipelineStatus.ERROR:
                if self.pbar is not None:
                    self.pbar.close()
                    self.pbar = None
                print(f"\nError: {self.state.error}")
                return
            elif self.state.status == PipelineStatus.ABORTED:
                if self.pbar is not None:
                    self.pbar.close()
                    self.pbar = None
                print(f"\nPipeline aborted: {self.state.error}")
                return
            elif self.state.status == PipelineStatus.COMPLETED:
                if self.pbar is not None:
                    self.pbar.close()
                    self.pbar = None
                print("\nPipeline completed successfully")
                return
            elif self.state.status == PipelineStatus.INITIALIZING:
                print(f"\nInitializing pipeline...")
            elif self.state.status == PipelineStatus.RUNNING:
                print(f"\nPipeline started")
            elif self.state.status == PipelineStatus.INTERRUPTED:
                if self.pbar is not None:
                    self.pbar.close()
                    self.pbar = None
                print(f"\nPipeline interrupted")
                return

        # Handle step changes
        if self.state.current_step != self.last_step and self.state.current_step:
            self.last_step = self.state.current_step
            # Close existing progress bar when step changes
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None
            
            if self.state.step_details:
                print(f"\n{self.state.step_details}")

        # Display warnings
        if self.state.warning and self.state.warning != self.previous_state.warning:
            print(f"\nWarning: {self.state.warning}")
        
        # Display messages
        if self.state.message and self.state.message != self.previous_state.message:
            # For batch operations, show message without newline to avoid cluttering
            if "batch" in self.state.message.lower() or "waiting" in self.state.message.lower():
                if self.pbar is None:
                    print(f"{self.state.message}")
            else:
                print(f"\n{self.state.message}")

    async def _abort(self):
        """Abort the pipeline and close the progress bar."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        print("\nPipeline aborted")
        sys.exit(1)