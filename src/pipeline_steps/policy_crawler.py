import os
import time
import json
import threading
import urllib.parse
import random
import requests
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.state_manager import BaseStateManager


# Custom exception classes
class PDFFileException(Exception):
    """Exception raised when a PDF file is encountered instead of a web page."""
    pass


class NoPolicyException(Exception):
    """Exception raised when no privacy policy is found on the web page."""
    pass


class EmptyPolicyException(Exception):
    """Exception raised when the privacy policy is empty."""
    pass


class InvalidURLException(Exception):
    """Exception raised when the URL is invalid or malformed."""
    pass


class AntiBotManager:
    """Manages anti-bot detection and evasion techniques."""
    
    def __init__(self, use_proxies: bool = False, proxy_list: List[str] = None):
        self.use_proxies = use_proxies
        self.proxy_list = proxy_list or []
        self.current_proxy = None
        
        # Realistic user agents for different browsers
        self.user_agents = [
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            
            # Firefox on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
            
            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            
            # Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            
            # Chrome on Linux
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
        
        # Screen resolutions for realistic viewport
        self.screen_resolutions = [
            (1920, 1080), (1366, 768), (1440, 900), (1536, 864),
            (1280, 720), (1600, 900), (1024, 768), (2560, 1440)
        ]
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(self.user_agents)
    
    def get_random_screen_resolution(self) -> tuple:
        """Get a random screen resolution."""
        return random.choice(self.screen_resolutions)
    
    def get_random_proxy(self) -> Optional[str]:
        """Get a random proxy from the list."""
        if not self.use_proxies or not self.proxy_list:
            return None
        return random.choice(self.proxy_list)
    
    def rotate_proxy(self):
        """Rotate to a different proxy."""
        if self.proxy_list:
            self.current_proxy = self.get_random_proxy()
    
    def get_stealth_options(self) -> Dict[str, Any]:
        """Get stealth options for WebDriver setup."""
        width, height = self.get_random_screen_resolution()
        
        return {
            'user_agent': self.get_random_user_agent(),
            'screen_width': width,
            'screen_height': height,
            'proxy': self.get_random_proxy() if self.use_proxies else None,
            'timezone': random.choice(['America/New_York', 'Europe/London', 'Asia/Tokyo', 'Australia/Sydney']),
            'language': random.choice(['en-US', 'en-GB', 'en-CA', 'en-AU']),
        }


class DriverPool:
    """Simple driver pool for reusing WebDriver instances."""
    
    def __init__(self, max_drivers=3):
        self.max_drivers = max_drivers
        self.drivers = []
        self.lock = threading.Lock()
    
    def get_driver(self, setup_func):
        """Get a driver from the pool or create a new one."""
        with self.lock:
            if self.drivers:
                driver = self.drivers.pop()
                # Test if driver is still responsive
                try:
                    driver.current_url
                    return driver
                except:
                    # Driver is dead, create a new one
                    pass
            return setup_func()
    
    def return_driver(self, driver):
        """Return a driver to the pool."""
        with self.lock:
            if len(self.drivers) < self.max_drivers:
                try:
                    # Clear cookies and cache to prepare for next use
                    driver.delete_all_cookies()
                    self.drivers.append(driver)
                except:
                    # Driver is dead, don't return it
                    try:
                        driver.quit()
                    except:
                        pass
            else:
                try:
                    driver.quit()
                except:
                    pass


class PolicyCrawler(PipelineStep):
    """Crawler step that reads metadata JSON files and fetches privacy policies."""

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
            crawl_retries: int = 2,
            use_proxies: bool = False,
            proxy_list: List[str] = None,
            enable_javascript: bool = True,
            stealth_mode: bool = True,
            request_delay: float = 1.0,
            max_concurrent_requests: int = 3
    ):
        super().__init__(
            run_id=run_id,
            task='policy_crawl',
            details='Fetch privacy policy from metadata URL',
            skip=skip,
            is_llm_step=False,  # Policy crawler is not an LLM step
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
        self.enable_javascript = enable_javascript
        self.stealth_mode = stealth_mode
        self.request_delay = request_delay
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize anti-bot manager
        self.anti_bot_manager = AntiBotManager(use_proxies=use_proxies, proxy_list=proxy_list)
        
        # Enhanced driver pool with concurrency control
        self.driver_pool = DriverPool(max_drivers=max_concurrent_requests)
        self._cache = {}  # Simple in-memory cache for policies
        
        # Request rate limiting
        self.last_request_time = 0
        self.request_lock = threading.Lock()

    async def execute(self, pkg: str):
        """Execute the policy crawling step for a single package."""
        try:
            # Rate limiting
            with self.request_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.request_delay:
                    sleep_time = self.request_delay - time_since_last
                    print(f"Rate limiting: waiting {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                self.last_request_time = time.time()
            
            print(f"Fetching privacy policy for package {pkg}...")
            
            # Read the metadata JSON file
            metadata_file = f"{self.in_folder}/{pkg}.json"
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
            with open(metadata_file, 'r') as f:
                app_metadata = json.load(f)
            
            # Extract privacy policy URL from metadata
            privacy_policy_url = app_metadata.get('privacy_policy_url')
            if not privacy_policy_url:
                raise NoPolicyException(f"No privacy policy URL found for package {pkg}")
            
            # Validate and sanitize URL
            validated_url = self._validate_url(privacy_policy_url)
            print(f"Loading privacy policy from: {validated_url}")
            
            # Check cache first
            if validated_url in self._cache:
                print(f"Using cached policy for {pkg}")
                privacy_policy_html = self._cache[validated_url]
            else:
                # Fetch the privacy policy with retry logic
                privacy_policy_html = self._fetch_with_retry(validated_url)
                if privacy_policy_html:
                    self._cache[validated_url] = privacy_policy_html
            
            if not privacy_policy_html:
                raise EmptyPolicyException(f"Could not fetch privacy policy for package {pkg}")
            
            # Save the privacy policy document
            policy_file = f"{self.out_folder}/{pkg}.html"
            util.write_to_file(policy_file, privacy_policy_html)
            
            print(f"Successfully fetched privacy policy for package {pkg}")
            await self.state_manager.update_state(file_progress=1.0)
            
        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))
            # Create log directory if it doesn't exist
            log_dir = f"../../output/{self.run_id}/log"
            os.makedirs(log_dir, exist_ok=True)
            util.write_to_file(f"{log_dir}/failed_policy_crawl.txt", pkg)
            return None

    async def prepare_batch(self, pkg: str):
        """Prepare batch entry for policy crawling. Since policy crawler is not an LLM step, this returns None."""
        # Policy crawler doesn't use batch processing since it's not an LLM step
        return None

    def _validate_url(self, url: str) -> str:
        """Validate and sanitize URL."""
        if not url or not url.strip():
            raise InvalidURLException("URL cannot be empty")
        
        url = url.strip()
        
        # Handle relative URLs by assuming HTTPS
        if url.startswith('/'):
            # Extract domain from the original URL if available
            url = f"https://example.com{url}"  # Fallback domain
        elif not url.startswith(('http://', 'https://')):
            # Assume HTTPS if no protocol specified
            url = f"https://{url}"
        
        # Basic URL validation
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise InvalidURLException(f"Invalid URL structure: {url}")
        except Exception as e:
            raise InvalidURLException(f"URL parsing failed: {url} - {e}")
        
        return url

    def _fetch_with_retry(self, url: str, max_retries: int = None) -> str:
        """Fetch content with retry logic, exponential backoff, and enhanced error handling."""
        if max_retries is None:
            max_retries = self.crawl_retries
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Rotate proxy on retry if available
                if attempt > 0 and self.anti_bot_manager.use_proxies:
                    print(f"Rotating proxy for attempt {attempt + 1}...")
                    self.anti_bot_manager.rotate_proxy()
                
                return self._fetch_privacy_policy(url)
            except Exception as e:
                last_exception = e
                
                # Handle specific error types
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"Timeout error on attempt {attempt + 1}, retrying...")
                elif "connection" in str(e).lower() or "network" in str(e).lower():
                    print(f"Network error on attempt {attempt + 1}, retrying...")
                elif "forbidden" in str(e).lower() or "403" in str(e).lower():
                    print(f"Access forbidden on attempt {attempt + 1}, rotating proxy and retrying...")
                    if self.anti_bot_manager.use_proxies:
                        self.anti_bot_manager.rotate_proxy()
                elif "too many requests" in str(e).lower() or "429" in str(e).lower():
                    print(f"Rate limited on attempt {attempt + 1}, waiting longer...")
                    wait_time = min(2 ** (attempt + 2), 60)  # Longer wait for rate limits
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                
                if attempt == max_retries - 1:
                    print(f"All {max_retries} attempts failed for {url}")
                    raise last_exception
                
                # Exponential backoff with jitter
                base_wait = 2 ** attempt
                jitter = random.uniform(0.5, 1.5)
                wait_time = min(base_wait * jitter, 30)
                print(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        return ''

    def _fetch_privacy_policy(self, privacy_policy_url: str) -> str:
        """Fetch the privacy policy from the given URL with enhanced capabilities."""
        driver = None
        
        try:
            # Get driver from pool
            driver = self.driver_pool.get_driver(self._setup_driver)
            
            # Set up wait with shorter timeout for better responsiveness
            wait = WebDriverWait(driver, 15)

            print('Loading privacy policy from URL: %s...' % privacy_policy_url)

            # Navigate to the URL
            driver.get(privacy_policy_url)
            
            # Apply stealth measures after page load
            if self.stealth_mode:
                self._apply_page_stealth_measures(driver)
                # Test stealth measures (optional - can be disabled for performance)
                self._test_stealth_measures(driver)

            # Handle special page types
            page_source = self._handle_special_page_types(driver, privacy_policy_url)

            # Smart waiting for page content
            self._wait_for_content(driver, wait)

            # Check for forwarding notice and follow it
            if self._forwarding_notice_present(page_source):
                try:
                    # Look for actual links, not just text
                    links = driver.find_elements(By.TAG_NAME, 'a')
                    for link in links:
                        href = link.get_attribute('href')
                        if href and href.startswith('http'):
                            print('Forwarding notice found, following it to %s...' % href)
                            driver.get(href)
                            # Apply stealth measures after redirect
                            if self.stealth_mode:
                                self._apply_page_stealth_measures(driver)
                            # Wait for the redirected page to load
                            self._wait_for_content(driver, wait)
                            # Handle special page types again
                            page_source = self._handle_special_page_types(driver, href)
                            break
                    else:
                        print('Forwarding notice found but no valid link detected, proceeding...')
                except Exception as e:
                    print(f'Error following forwarding notice: {e}, proceeding...')

            # Accept cookies with enhanced detection
            self._accept_all_cookies_enhanced(driver)

            print('Done loading privacy policy page.')

            # Extract the HTML source from the page
            page_source = driver.page_source
            
            # Extract main content (optional - can be disabled for full page capture)
            if self.stealth_mode:
                try:
                    page_source = self._extract_main_content(page_source)
                    print("Extracted main content from page")
                except Exception as e:
                    print(f"Content extraction failed, using full page: {e}")

            # Return driver to pool
            self.driver_pool.return_driver(driver)
            driver = None

            return page_source
            
        except Exception as e:
            print(f"Error fetching privacy policy: {e}")
            if driver:
                try:
                    self.driver_pool.return_driver(driver)
                except:
                    pass
            raise e

    def _wait_for_content(self, driver: WebDriver, wait: WebDriverWait):
        """Smart waiting for page content to load."""
        try:
            # Wait for body to be present
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            
            # Wait for some meaningful content to be loaded
            def has_content(d):
                try:
                    body = d.find_element(By.TAG_NAME, 'body')
                    text = body.text.strip()
                    return len(text) > 50
                except:
                    return False
            
            wait.until(has_content)
            
            # Additional short wait for dynamic content
            time.sleep(1)
            
        except TimeoutException:
            print("Warning: Page content loading timeout, proceeding anyway...")

    def _setup_driver(self, driver_timeout=15, headless=True, page_load_strategy='eager'):
        """Set up Selenium WebDriver with optimized settings and anti-bot detection."""
        print('Setting up Selenium WebDriver...')
        options = Options()

        # Get stealth options
        stealth_options = self.anti_bot_manager.get_stealth_options()
        
        # Language and localization
        options.add_argument(f'--lang={stealth_options["language"]}')
        options.add_argument(f"--accept-language={stealth_options['language']},*")
        
        # User agent rotation
        if self.stealth_mode:
            options.add_argument(f'--user-agent={stealth_options["user_agent"]}')
        
        # Screen resolution for realistic viewport
        if self.stealth_mode:
            width, height = stealth_options["screen_width"], stealth_options["screen_height"]
            options.add_argument(f'--window-size={width},{height}')
        
        # Selenium Stealth Measures - Critical for avoiding detection
        if self.stealth_mode:
            # Remove Selenium-specific properties (Firefox-compatible)
            options.add_argument("--disable-blink-features=AutomationControlled")
            
            # Disable webdriver mode
            options.add_argument("--disable-web-security")
            options.add_argument("--disable-features=VizDisplayCompositor")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-gpu")
            
            # Remove automation indicators
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-plugins")
            options.add_argument("--disable-logging")
            
            # Additional stealth measures
            options.add_argument("--disable-blink-features")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-web-security")
            options.add_argument("--allow-running-insecure-content")
            options.add_argument("--disable-features=VizDisplayCompositor")
            options.add_argument("--disable-features=TranslateUI")
            options.add_argument("--disable-ipc-flooding-protection")
            
            # Firefox-specific stealth options
            options.add_argument("--disable-default-apps")
            options.add_argument("--disable-sync")
            options.add_argument("--disable-background-timer-throttling")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument("--disable-renderer-backgrounding")
            options.add_argument("--disable-features=TranslateUI")
            options.add_argument("--disable-ipc-flooding-protection")
            options.add_argument("--disable-features=VizDisplayCompositor")
            options.add_argument("--disable-features=TranslateUI")
            options.add_argument("--disable-ipc-flooding-protection")
            
            # Remove navigator.webdriver property
            options.add_argument("--disable-web-security")
            options.add_argument("--disable-features=VizDisplayCompositor")
        
        # Performance optimizations (conditional based on stealth mode)
        if not self.stealth_mode:
            options.add_argument("--disable-images")
            options.add_argument("--disable-css")
            options.add_argument("--disable-fonts")
        
        # JavaScript handling - enable by default for modern sites
        if not self.enable_javascript:
            options.add_argument("--disable-javascript")
        
        # Always disable these for stability
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-logging")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")

        # Cookie and popup handling
        options.add_argument("--cookie-policy=accept-all")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-infobars")

        # Memory and resource management
        options.add_argument("--memory-pressure-off")
        options.add_argument("--max_old_space_size=4096")
        
        # Proxy support
        if stealth_options["proxy"]:
            options.add_argument(f'--proxy-server={stealth_options["proxy"]}')

        if headless:
            options.add_argument('-headless')

        options.page_load_strategy = page_load_strategy

        driver = webdriver.Firefox(options=options)
        driver.set_page_load_timeout(driver_timeout)
        driver.set_script_timeout(driver_timeout)
        
        # Set viewport size for consistency
        if self.stealth_mode:
            width, height = stealth_options["screen_width"], stealth_options["screen_height"]
            driver.set_window_size(width, height)
        
        # Apply stealth JavaScript to remove automation indicators
        if self.stealth_mode:
            self._apply_stealth_scripts(driver)

        print('Driver setup complete.')
        return driver

    def _apply_stealth_scripts(self, driver):
        """Apply JavaScript stealth measures to remove automation indicators."""
        try:
            # Remove navigator.webdriver property
            driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            # Remove automation-related properties
            driver.execute_script("""
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
            """)
            
            # Override permissions API
            driver.execute_script("""
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)
            
            # Override plugins
            driver.execute_script("""
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
            """)
            
            # Override languages
            driver.execute_script("""
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
            """)
            
            # Override connection
            driver.execute_script("""
                Object.defineProperty(navigator, 'connection', {
                    get: () => ({
                        effectiveType: '4g',
                        rtt: 50,
                        downlink: 10,
                        saveData: false
                    }),
                });
            """)
            
            # Override chrome object
            driver.execute_script("""
                window.chrome = {
                    runtime: {},
                };
            """)
            
            # Override permissions
            driver.execute_script("""
                const originalQuery = window.navigator.permissions.query;
                return originalQuery = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)
            
            # Remove automation-related CSS
            driver.execute_script("""
                const style = document.createElement('style');
                style.innerHTML = `
                    * {
                        -webkit-user-select: auto !important;
                        -moz-user-select: auto !important;
                        -ms-user-select: auto !important;
                        user-select: auto !important;
                    }
                `;
                document.head.appendChild(style);
            """)
            
            print("Applied stealth scripts successfully")
            
        except Exception as e:
            print(f"Warning: Some stealth scripts failed to apply: {e}")

    def _accept_all_cookies_enhanced(self, driver):
        """Enhanced cookie banner detection and handling with human-like behavior."""
        # Wait a bit before looking for cookie banners (human-like)
        time.sleep(random.uniform(0.5, 2.0))
        
        cookie_patterns = [
            '//div[contains(@class, "cookie")]',
            '//div[contains(@id, "cookie")]',
            '//div[contains(@class, "consent")]',
            '//div[contains(@class, "gdpr")]',
            '//div[contains(@class, "banner")]',
            '//div[contains(@class, "notice")]',
            '//div[contains(@class, "popup")]',
            '//div[contains(@class, "modal")]',
            '//div[contains(@class, "overlay")]',
            '//div[contains(@class, "dialog")]',
            '//div[contains(@class, "privacy")]',
            '//div[contains(@class, "terms")]',
            '//div[contains(@class, "legal")]',
            '//div[contains(@class, "notification")]',
            '//div[contains(@class, "alert")]',
            '//div[contains(@class, "message")]',
            '//div[contains(@class, "toast")]',
            '//div[contains(@class, "snackbar")]',
            '//div[contains(@class, "disclaimer")]',
            '//div[contains(@class, "compliance")]'
        ]
        
        accept_texts = [
            'Accept', 'Akzeptieren', 'Accept All', 'Allow All', 'OK', 'Got it',
            'I Accept', 'Agree', 'Continue', 'Proceed', 'Allow', 'Enable',
            'Accept Cookies', 'Accept All Cookies', 'Allow Cookies', 'Yes',
            'Confirm', 'Understood', 'I Understand', 'Close', 'Dismiss',
            'Accept & Continue', 'Accept and Continue', 'I Agree',
            'Accept All Cookies', 'Allow All Cookies', 'Accept & Close'
        ]
        
        # Try multiple patterns and button texts
        for pattern in cookie_patterns:
            try:
                cookie_banners = driver.find_elements(By.XPATH, pattern)
                for banner in cookie_banners:
                    for text in accept_texts:
                        try:
                            # Try different button selectors
                            selectors = [
                                f'.//button[contains(text(), "{text}")]',
                                f'.//a[contains(text(), "{text}")]',
                                f'.//span[contains(text(), "{text}")]',
                                f'.//div[contains(text(), "{text}")]',
                                f'.//input[@value="{text}"]',
                                f'.//*[contains(text(), "{text}")]'
                            ]
                            
                            for selector in selectors:
                                try:
                                    accept_button = banner.find_element(By.XPATH, selector)
                                    if accept_button.is_displayed() and accept_button.is_enabled():
                                        # Human-like behavior: move mouse to button first
                                        actions = ActionChains(driver)
                                        actions.move_to_element(accept_button)
                                        actions.pause(random.uniform(0.1, 0.3))
                                        actions.click()
                                        actions.perform()
                                        
                                        print(f"Clicked cookie accept button: {text}")
                                        time.sleep(random.uniform(0.3, 1.0))  # Human-like delay
                                        return
                                except:
                                    continue
                        except:
                            continue
            except:
                continue
        
        # If no cookie banner found, try to scroll and dismiss any overlays
        try:
            # Human-like scrolling
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(random.uniform(0.5, 1.5))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(0.5, 1.0))
        except:
            pass

    def _forwarding_notice_present(self, page_source):
        """
        Check if a forwarding notice is present on the page.

        Args:
            page_source (str): The HTML source of the page.

        Returns:
            bool: True if a forwarding notice is present, False otherwise.
        """
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Check for the specific forwarding notice class
        forwarding_notice = soup.find('div', class_='aXgaGb')
        if forwarding_notice:
            return True
        
        # Check for other common forwarding patterns
        forwarding_indicators = [
            'forwarding', 'redirect', 'you are being redirected',
            'click here to continue', 'proceed to'
        ]
        
        page_text = soup.get_text().lower()
        if any(indicator in page_text for indicator in forwarding_indicators):
            return True
        
        return False

    def _handle_special_page_types(self, driver: WebDriver, url: str) -> str:
        """Handle special page types like PDFs, redirects, and dynamic content."""
        current_url = driver.current_url
        
        # Handle PDF files
        if current_url.endswith('.pdf') or 'application/pdf' in driver.page_source:
            raise PDFFileException("PDF privacy policy not supported")
        
        # Handle redirects
        if current_url != url:
            print(f"Page redirected from {url} to {current_url}")
        
        # Handle JavaScript-heavy sites
        if self.enable_javascript:
            # Wait for dynamic content to load
            time.sleep(random.uniform(2, 4))
            
            # Try to scroll to trigger lazy loading
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(1, 2))
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(random.uniform(0.5, 1))
            except:
                pass
        
        return driver.page_source

    def _extract_main_content(self, html_content: str) -> str:
        """Extract main content from HTML, removing ads and navigation."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove common non-content elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Remove common ad and tracking elements
        ad_selectors = [
            '[class*="ad"]', '[class*="ads"]', '[class*="advertisement"]',
            '[id*="ad"]', '[id*="ads"]', '[id*="advertisement"]',
            '[class*="tracking"]', '[class*="analytics"]',
            '[class*="social"]', '[class*="share"]',
            '[class*="cookie"]', '[class*="banner"]',
            '[class*="popup"]', '[class*="modal"]'
        ]
        
        for selector in ad_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Common main content selectors
        content_selectors = [
            'main', '[role="main"]', '.main', '#main',
            '.content', '#content', '.post-content', '.article-content',
            '.privacy-content', '.legal-content', '.policy-content',
            '.document-content', '.page-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body')
        
        return str(main_content) if main_content else html_content 

    def _apply_page_stealth_measures(self, driver):
        """Apply stealth measures after each page load to maintain stealth."""
        try:
            # Remove any automation indicators that might have been added
            driver.execute_script("""
                // Remove webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Remove automation-related properties
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
                
                // Override webdriver property in different ways
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });
                
                // Remove any selenium-specific properties
                delete window.$cdc_asdjflasutopfhvcZLmcfl_;
                delete window.$chrome_asyncScriptInfo;
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                
                // Override plugins to look more realistic
                Object.defineProperty(navigator, 'plugins', {
                    get: () => {
                        const plugins = [];
                        for (let i = 0; i < 3; i++) {
                            plugins.push({
                                name: `Plugin ${i}`,
                                filename: `plugin${i}.dll`,
                                description: `Plugin ${i} Description`,
                                length: 1
                            });
                        }
                        return plugins;
                    },
                });
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
                
                // Override connection
                Object.defineProperty(navigator, 'connection', {
                    get: () => ({
                        effectiveType: '4g',
                        rtt: 50,
                        downlink: 10,
                        saveData: false
                    }),
                });
                
                // Override chrome object
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {},
                    csi: function() {},
                    app: {}
                };
                
                // Override permissions API
                if (navigator.permissions) {
                    const originalQuery = navigator.permissions.query;
                    navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                }
                
                // Remove any automation-related CSS
                const style = document.createElement('style');
                style.innerHTML = `
                    * {
                        -webkit-user-select: auto !important;
                        -moz-user-select: auto !important;
                        -ms-user-select: auto !important;
                        user-select: auto !important;
                    }
                `;
                document.head.appendChild(style);
                
                // Override toString methods to hide automation
                const originalToString = Function.prototype.toString;
                Function.prototype.toString = function() {
                    if (this === navigator.webdriver) {
                        return 'function get webdriver() { [native code] }';
                    }
                    return originalToString.call(this);
                };
            """)
            
        except Exception as e:
            print(f"Warning: Page stealth measures failed to apply: {e}")

    def _test_stealth_measures(self, driver):
        """Test if stealth measures are working and detect automation indicators."""
        try:
            stealth_status = driver.execute_script("""
                const indicators = {};
                
                // Check for webdriver property
                indicators.webdriver = navigator.webdriver;
                
                // Check for automation-related properties
                indicators.cdc_properties = {
                    array: !!window.cdc_adoQpoasnfa76pfcZLmcfl_Array,
                    promise: !!window.cdc_adoQpoasnfa76pfcZLmcfl_Promise,
                    symbol: !!window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol
                };
                
                // Check for selenium-specific properties
                indicators.selenium_properties = {
                    cdc: !!window.$cdc_asdjflasutopfhvcZLmcfl_,
                    chrome_async: !!window.$chrome_asyncScriptInfo
                };
                
                // Check for automation-related CSS
                indicators.automation_css = document.querySelector('style[data-selenium]') !== null;
                
                // Check for automation-related attributes
                indicators.automation_attributes = document.querySelector('[data-selenium]') !== null;
                
                // Check for automation-related classes
                indicators.automation_classes = document.querySelector('.selenium') !== null;
                
                // Check for automation-related IDs
                indicators.automation_ids = document.querySelector('#selenium') !== null;
                
                // Check for automation-related scripts
                indicators.automation_scripts = Array.from(document.scripts).some(script => 
                    script.src.includes('selenium') || 
                    script.src.includes('webdriver') ||
                    script.textContent.includes('selenium') ||
                    script.textContent.includes('webdriver')
                );
                
                // Check for automation-related meta tags
                indicators.automation_meta = document.querySelector('meta[name*="selenium"]') !== null;
                
                // Check for automation-related iframes
                indicators.automation_iframes = Array.from(document.querySelectorAll('iframe')).some(iframe => 
                    iframe.src.includes('selenium') || 
                    iframe.src.includes('webdriver')
                );
                
                // Check for automation-related objects
                indicators.automation_objects = document.querySelector('object[data*="selenium"]') !== null;
                
                // Check for automation-related embeds
                indicators.automation_embeds = document.querySelector('embed[src*="selenium"]') !== null;
                
                // Check for automation-related applets
                indicators.automation_applets = document.querySelector('applet[code*="selenium"]') !== null;
                
                // Check for automation-related forms
                indicators.automation_forms = document.querySelector('form[action*="selenium"]') !== null;
                
                // Check for automation-related links
                indicators.automation_links = document.querySelector('a[href*="selenium"]') !== null;
                
                // Check for automation-related images
                indicators.automation_images = document.querySelector('img[src*="selenium"]') !== null;
                
                // Check for automation-related videos
                indicators.automation_videos = document.querySelector('video[src*="selenium"]') !== null;
                
                // Check for automation-related audios
                indicators.automation_audios = document.querySelector('audio[src*="selenium"]') !== null;
                
                // Check for automation-related canvases
                indicators.automation_canvases = document.querySelector('canvas[data-selenium]') !== null;
                
                // Check for automation-related svgs
                indicators.automation_svgs = document.querySelector('svg[data-selenium]') !== null;
                
                // Check for automation-related mathml
                indicators.automation_mathml = document.querySelector('math[data-selenium]') !== null;
                
                // Check for automation-related details
                indicators.automation_details = document.querySelector('details[data-selenium]') !== null;
                
                // Check for automation-related dialogs
                indicators.automation_dialogs = document.querySelector('dialog[data-selenium]') !== null;
                
                // Check for automation-related meters
                indicators.automation_meters = document.querySelector('meter[data-selenium]') !== null;
                
                // Check for automation-related progress
                indicators.automation_progress = document.querySelector('progress[data-selenium]') !== null;
                
                // Check for automation-related time
                indicators.automation_time = document.querySelector('time[data-selenium]') !== null;
                
                // Check for automation-related mark
                indicators.automation_mark = document.querySelector('mark[data-selenium]') !== null;
                
                // Check for automation-related ruby
                indicators.automation_ruby = document.querySelector('ruby[data-selenium]') !== null;
                
                // Check for automation-related rt
                indicators.automation_rt = document.querySelector('rt[data-selenium]') !== null;
                
                // Check for automation-related rp
                indicators.automation_rp = document.querySelector('rp[data-selenium]') !== null;
                
                // Check for automation-related bdi
                indicators.automation_bdi = document.querySelector('bdi[data-selenium]') !== null;
                
                // Check for automation-related bdo
                indicators.automation_bdo = document.querySelector('bdo[data-selenium]') !== null;
                
                // Check for automation-related wbr
                indicators.automation_wbr = document.querySelector('wbr[data-selenium]') !== null;
                
                // Check for automation-related picture
                indicators.automation_picture = document.querySelector('picture[data-selenium]') !== null;
                
                // Check for automation-related source
                indicators.automation_source = document.querySelector('source[data-selenium]') !== null;
                
                // Check for automation-related track
                indicators.automation_track = document.querySelector('track[data-selenium]') !== null;
                
                // Check for automation-related map
                indicators.automation_map = document.querySelector('map[data-selenium]') !== null;
                
                // Check for automation-related area
                indicators.automation_area = document.querySelector('area[data-selenium]') !== null;
                
                // Check for automation-related table
                indicators.automation_table = document.querySelector('table[data-selenium]') !== null;
                
                // Check for automation-related caption
                indicators.automation_caption = document.querySelector('caption[data-selenium]') !== null;
                
                // Check for automation-related colgroup
                indicators.automation_colgroup = document.querySelector('colgroup[data-selenium]') !== null;
                
                // Check for automation-related col
                indicators.automation_col = document.querySelector('col[data-selenium]') !== null;
                
                // Check for automation-related tbody
                indicators.automation_tbody = document.querySelector('tbody[data-selenium]') !== null;
                
                // Check for automation-related thead
                indicators.automation_thead = document.querySelector('thead[data-selenium]') !== null;
                
                // Check for automation-related tfoot
                indicators.automation_tfoot = document.querySelector('tfoot[data-selenium]') !== null;
                
                // Check for automation-related tr
                indicators.automation_tr = document.querySelector('tr[data-selenium]') !== null;
                
                // Check for automation-related td
                indicators.automation_td = document.querySelector('td[data-selenium]') !== null;
                
                // Check for automation-related th
                indicators.automation_th = document.querySelector('th[data-selenium]') !== null;
                
                // Check for automation-related fieldset
                indicators.automation_fieldset = document.querySelector('fieldset[data-selenium]') !== null;
                
                // Check for automation-related legend
                indicators.automation_legend = document.querySelector('legend[data-selenium]') !== null;
                
                // Check for automation-related label
                indicators.automation_label = document.querySelector('label[data-selenium]') !== null;
                
                // Check for automation-related input
                indicators.automation_input = document.querySelector('input[data-selenium]') !== null;
                
                // Check for automation-related button
                indicators.automation_button = document.querySelector('button[data-selenium]') !== null;
                
                // Check for automation-related select
                indicators.automation_select = document.querySelector('select[data-selenium]') !== null;
                
                // Check for automation-related datalist
                indicators.automation_datalist = document.querySelector('datalist[data-selenium]') !== null;
                
                // Check for automation-related optgroup
                indicators.automation_optgroup = document.querySelector('optgroup[data-selenium]') !== null;
                
                // Check for automation-related option
                indicators.automation_option = document.querySelector('option[data-selenium]') !== null;
                
                // Check for automation-related textarea
                indicators.automation_textarea = document.querySelector('textarea[data-selenium]') !== null;
                
                // Check for automation-related output
                indicators.automation_output = document.querySelector('output[data-selenium]') !== null;
                
                // Check for automation-related progress
                indicators.automation_progress = document.querySelector('progress[data-selenium]') !== null;
                
                // Check for automation-related meter
                indicators.automation_meter = document.querySelector('meter[data-selenium]') !== null;
                
                // Check for automation-related keygen
                indicators.automation_keygen = document.querySelector('keygen[data-selenium]') !== null;
                
                return indicators;
            """)
            
            # Check if any automation indicators were found
            automation_detected = any([
                stealth_status.get('webdriver'),
                any(stealth_status.get('cdc_properties', {}).values()),
                any(stealth_status.get('selenium_properties', {}).values()),
                stealth_status.get('automation_css'),
                stealth_status.get('automation_attributes'),
                stealth_status.get('automation_classes'),
                stealth_status.get('automation_ids'),
                stealth_status.get('automation_scripts'),
                stealth_status.get('automation_meta'),
                stealth_status.get('automation_iframes'),
                stealth_status.get('automation_objects'),
                stealth_status.get('automation_embeds'),
                stealth_status.get('automation_applets'),
                stealth_status.get('automation_forms'),
                stealth_status.get('automation_links'),
                stealth_status.get('automation_images'),
                stealth_status.get('automation_videos'),
                stealth_status.get('automation_audios'),
                stealth_status.get('automation_canvases'),
                stealth_status.get('automation_svgs'),
                stealth_status.get('automation_mathml'),
                stealth_status.get('automation_details'),
                stealth_status.get('automation_dialogs'),
                stealth_status.get('automation_meters'),
                stealth_status.get('automation_progress'),
                stealth_status.get('automation_time'),
                stealth_status.get('automation_mark'),
                stealth_status.get('automation_ruby'),
                stealth_status.get('automation_rt'),
                stealth_status.get('automation_rp'),
                stealth_status.get('automation_bdi'),
                stealth_status.get('automation_bdo'),
                stealth_status.get('automation_wbr'),
                stealth_status.get('automation_picture'),
                stealth_status.get('automation_source'),
                stealth_status.get('automation_track'),
                stealth_status.get('automation_map'),
                stealth_status.get('automation_area'),
                stealth_status.get('automation_table'),
                stealth_status.get('automation_caption'),
                stealth_status.get('automation_colgroup'),
                stealth_status.get('automation_col'),
                stealth_status.get('automation_tbody'),
                stealth_status.get('automation_thead'),
                stealth_status.get('automation_tfoot'),
                stealth_status.get('automation_tr'),
                stealth_status.get('automation_td'),
                stealth_status.get('automation_th'),
                stealth_status.get('automation_fieldset'),
                stealth_status.get('automation_legend'),
                stealth_status.get('automation_label'),
                stealth_status.get('automation_input'),
                stealth_status.get('automation_button'),
                stealth_status.get('automation_select'),
                stealth_status.get('automation_datalist'),
                stealth_status.get('automation_optgroup'),
                stealth_status.get('automation_option'),
                stealth_status.get('automation_textarea'),
                stealth_status.get('automation_output'),
                stealth_status.get('automation_keygen')
            ])
            
            if automation_detected:
                print("⚠️  WARNING: Automation indicators detected! Stealth measures may not be working properly.")
                print(f"   Indicators found: {stealth_status}")
            else:
                print("✅ Stealth measures working correctly - no automation indicators detected")
                
            return not automation_detected
            
        except Exception as e:
            print(f"Warning: Could not test stealth measures: {e}")
            return False 