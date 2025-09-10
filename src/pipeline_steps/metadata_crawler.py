import os
import time
import json

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.pipeline_steps.playstore_parser import PlayStoreParser
from src.state_manager import BaseStateManager


# Custom exception classes
class NotInPlayStoreException(Exception):
    """Exception raised when the app is not found in the Google Play Store."""
    pass


class MetadataCrawler(PipelineStep):
    """Crawler step that fetches Play Store pages and extracts app metadata."""

    def __init__(
            self,
            run_id: str,
            skip: bool,
            in_folder: str,
            out_folder: str,
            state_manager: BaseStateManager,
            batch_input_file: str = "batch_input.json",
            batch_metadata_file: str = "batch_metadata.json",
            batch_results_file: str = "batch_results.jsonl",
            batch_errors_file: str = "batch_errors.jsonl",
            is_batch_step: bool = False,
            parallel_prompt: bool = False,
            model: str = None,
            client: ApiBase = None,
            crawl_retries: int = 2
    ):
        super().__init__(
            run_id=run_id,
            task='metadata_crawl',
            details='Fetch Play Store page and extract app metadata',
            skip=skip,
            is_llm_step=False,  # Metadata crawler is not an LLM step
            is_batch_step=is_batch_step,
            in_folder=in_folder,
            out_folder=out_folder,
            state_manager=state_manager,
            batch_input_file=batch_input_file,
            batch_metadata_file=batch_metadata_file,
            batch_results_file=batch_results_file,
            batch_errors_file=batch_errors_file,
            parallel_prompt=parallel_prompt,
            model=model,
            client=client
        )
        self.crawl_retries = crawl_retries

    async def execute(self, pkg: str):
        """Execute the metadata crawling step for a single package."""
        try:
            print(f"Fetching metadata for package {pkg}...")
            
            # Fetch the Play Store page for the given package name
            playstore_html = self._fetch_playstore_page(pkg, self.crawl_retries)
            if not playstore_html:
                raise NotInPlayStoreException(f"Could not fetch Play Store page for package {pkg}")
            
            # Extract app metadata using the PlayStore parser
            app_metadata = self._extract_app_metadata(playstore_html)
            if not app_metadata:
                raise Exception(f"Could not extract app metadata for package {pkg}")
            
            # Save the extracted metadata to JSONL file
            metadata_file = f"{self.out_folder}/{pkg}.jsonl"
            util.append_to_file(metadata_file, json.dumps(app_metadata))
            
            print(f"Successfully extracted metadata for package {pkg}")
            await self.state_manager.update_state(file_progress=1.0)
            
        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))
            # Create log directory if it doesn't exist
            log_dir = f"../../output/{self.run_id}/log"
            os.makedirs(log_dir, exist_ok=True)
            util.write_to_file(f"{log_dir}/failed_metadata_crawl.txt", pkg)
            return None

    async def prepare_batch(self, pkg: str):
        """Prepare batch entry for metadata crawling. Since metadata crawler is not an LLM step, this returns None."""
        # Metadata crawler doesn't use batch processing since it's not an LLM step
        return None

    def _fetch_playstore_page(self, pkg_name: str, retries: int = 0, driver: WebDriver = None) -> str:
        """Fetch the Play Store page for the given app package name."""
        kill_driver = False

        try:
            # Start driver
            if driver is None:
                kill_driver = True
                driver = self._setup_driver(driver_timeout=30, page_load_strategy='normal')

            # Open play store for the given app package name
            url = "https://play.google.com/store/apps/details?id="
            wait = WebDriverWait(driver, 10)

            print('Loading PlayStore page %s%s...' % (url, pkg_name))

            driver.get(f'{url}{pkg_name}')

            # Wait for page to load and find body
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body")))
            if 'the requested URL was not found on this server' in driver.page_source:
                print('App not available in PlayStore')
                raise NotInPlayStoreException

            print('Accepting all cookies on Play Store page...')
            self._accept_all_cookies(driver)

            print('Done loading Play Store page.')

            # Extract the HTML source from the page
            page = driver.page_source

            # Perform clean-up by killing the driver if necessary
            if kill_driver:
                driver.quit()

            return page
        except Exception as e:
            print(f"Error fetching Play Store page: {e}")
            if kill_driver and driver:
                driver.quit()
            return ''

    def _extract_app_metadata(self, playstore_html: str) -> dict:
        """Extract app metadata from the Play Store page using the PlayStore parser."""
        try:
            # Create a temporary PlayStore parser instance
            temp_parser = PlayStoreParser(
                run_id='metadata_crawler_temp',
                skip=False,
                in_folder='.',
                out_folder='.',
                state_manager=None
            )
            
            # Parse the Play Store HTML
            metadata = temp_parser._parse_playstore_html(playstore_html)
            print(f"Extracted metadata with {len(metadata)} fields")
            return metadata
        except Exception as e:
            print(f"Error extracting app metadata: {e}")
            return {}

    def _setup_driver(self, driver_timeout=10, headless=True, page_load_strategy='eager'):
        """Set up Selenium WebDriver."""
        print('Setting up Selenium WebDriver...')
        options = Options()

        options.add_argument('--lang=en-US')  # Set browser language to English
        options.add_argument("--cookie-policy=accept-all")
        options.add_argument("--enable-javascript")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-infobars")
        # Disable images
        options.add_argument("--blink-settings=imagesEnabled=false")
        # Set the Accept-Language header
        options.add_argument("--accept-language=en,*")  # Set Accept-Language to accept all English languages

        if headless:
            options.add_argument('-headless')  # use headless mode to avoid constant browser popups
        options.page_load_strategy = page_load_strategy  # allow timeouts before the page has fully loaded

        driver = webdriver.Firefox(options=options)
        driver.set_page_load_timeout(driver_timeout)

        print('Driver setup complete.')
        return driver

    def _accept_all_cookies(self, driver):
        """Accept all cookies on the page."""
        try:
            # Find the cookie banner element
            cookie_banner = driver.find_element(By.XPATH,
                                                '//div[contains(@class, "cookie") or contains(@id, "banner") or contains(@class, "banner") or contains(@id, "banner") or contains(@class, "consent") or contains(@id, "consent") or contains(@class, "notice") or contains(@class, "policy") or contains(@class, "message") or contains(@class, "modal") or contains(@class, "popup") or contains(@id, "popup") or contains(@class, "accept") or contains(@id, "accept")]')

            # Try to find and click an "Accept" button
            accept_button = None
            try:
                accept_button = cookie_banner.find_element(By.XPATH, './/button[contains(text(), "Accept")]')
            except:
                pass
            try:
                accept_button = cookie_banner.find_element(By.XPATH, './/button[contains(text(), "Akzeptieren")]')
            except:
                pass

            if accept_button is not None:
                accept_button.click()
            else:
                # Dismiss the cookie banner by clicking outside or scrolling
                driver.execute_script("arguments[0].scrollIntoView(true);", cookie_banner)
        except Exception as e:
            pass 