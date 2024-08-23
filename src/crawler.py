import csv
import logging
import os
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src import util


def execute(id_file: str, out_folder: str, crawl_retries: int):
    if id_file is not None and os.path.exists(id_file):
        if id_file.endswith('.csv'):
            pkgs = fetch_ids_in_csv(id_file)
        else:
            pkgs = fetch_ids_in_file(id_file)

        crawl_list(pkgs, out_folder, crawl_retries)
    else:
        print('Invalid input file path.')


def fetch_ids_in_file(id_file: str):
    pkgs = []
    with open(id_file, 'r') as file:
        for line in file:
            pkgs.append(line.strip())
    return pkgs


def fetch_ids_in_csv(id_file: str):
    pkgs = []
    with open(id_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            pkgs.append(row[0])
    return pkgs


# Custom exception classes
class PDFFileException(Exception):
    """Exception raised when a PDF file is encountered instead of a web page."""

    pass


class LanguageException(Exception):
    """Exception raised when the language of the web page is not English."""

    pass


class NotInPlayStoreException(Exception):
    """Exception raised when the app is not found in the Google Play Store."""

    pass


class EmptyPolicyException(Exception):
    """Exception raised when the privacy policy is empty."""

    pass


class AccessDeniedException(Exception):
    """Exception raised when access to the web page is denied."""

    pass


class NoPolicyException(Exception):
    """Exception raised when no privacy policy is found on the web page."""

    pass


class PageNotFoundException(Exception):
    """Exception raised when no page is found (404)."""

    pass


def setup_driver(driver_timeout=10, headless=True, page_load_strategy='eager'):
    logging.debug('Setting up Selenium WebDriver...')
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

    logging.debug('Setting up Firefox WebDriver...')
    driver = webdriver.Firefox(options=options)
    logging.debug('Successfully set up Firefox WebDriver.')

    driver.set_page_load_timeout(driver_timeout)

    print('Driver setup complete.')
    return driver


def crawl_list(pkgs: list[str], out_folder: str, retries: int = 2):
    for pkg in pkgs:
        policy = fetch_policy(pkg, retries)
        if len(policy) > 0:
            print('Saving original HTML to file...')
            file_name = "%s/%s.html" % (out_folder, pkg)
            util.write_to_file(file_name, policy)
        else:
            print('Empty original policy, not saving to file')
            continue


def fetch_policy(pkg_name: str, retries: int = 0, driver: WebDriver = None) -> str:
    """Get the original HTML of the privacy policy page for the given app package name.

    Args:
        pkg_name (str): The app's package name.
        retries (int, optional): The number of times to retry in case of timeouts or exceptions. Defaults to 0.
        driver (WebDriver, optional): The Selenium WebDriver to be used to fetch the policy page.

    Returns:
        str: The original HTML of the policy document.
    """
    kill_driver = False

    try:
        # Start driver
        if driver is None:
            kill_driver = True
            driver = setup_driver(driver_timeout=30, page_load_strategy='normal')

        # Open play store for the given app package name
        url = "https://play.google.com/store/apps/details?id="
        wait = WebDriverWait(driver, 10)

        print('Loading PlayStore page %s%s...' % (url, pkg_name))

        driver.get(f'{url}{pkg_name}')

        # Wait for page to load and find logo_url and app name
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body")))
        if 'the requested URL was not found on this server' in driver.page_source:
            print('App not available in PlayStore')
            raise NotInPlayStoreException

        # Expand the developers contact section
        xpath_expand = '//*[@id="developer-contacts-heading"]/div[2]/button'
        button_expand = driver.find_element(By.XPATH, xpath_expand)
        button_expand.click()

        # Check we don't have other windows open already
        assert len(driver.window_handles) == 1

        # Click the link which opens in a new window
        xpath_policy = '//*[@id="developer-contacts"]/div/div[last()]/div/a'
        wait.until(EC.visibility_of_element_located((By.XPATH, xpath_policy)))
        button_policy = driver.find_element(By.XPATH, xpath_policy)
        # Check if developer provides a policy, else throw exception
        if 'Privacy Policy' in button_policy.get_attribute('aria-label'):
            link_to_privacy_policy = button_policy.get_attribute('href')

            print('Loading policy from URL: %s...' % link_to_privacy_policy)

            driver.execute_script("window.open('" + link_to_privacy_policy + "', '_blank');")
            driver.switch_to.window(driver.window_handles[-1])
        else:
            raise NoPolicyException

        # Handle pdf policies
        is_pdf = driver.current_url.endswith(".pdf")
        if is_pdf:
            # TODO: implement
            raise PDFFileException

        # Wait for next page to load
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))
        time.sleep(2)  # Necessary as some content takes longer to load even after 'body' is visible

        # Check for forwarding notice
        if forwarding_notice_present(driver.page_source):
            link = driver.find_element(By.TAG_NAME, 'a')
            print('Forwarding notice found, following it to %s...' % link)
            driver.get(link.text)
            # Wait for next page to load
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))

        print('Accepting all cookies...')

        # Accept all cookies
        accept_all_cookies(driver)

        print('Done loading policy page.')

        # extract the HTML source from the page
        page = driver.page_source

        # perform clean-up by killing the driver if necessary
        if kill_driver:
            driver.quit()

        return page
    except Exception:
        return ''


def accept_all_cookies(driver):
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


def forwarding_notice_present(page_source):
    """
    Check if a forwarding notice is present on the page.

    Args:
        page_source (str): The HTML source of the page.

    Returns:
        bool: True if a forwarding notice is present, False otherwise.
    """
    soup = BeautifulSoup(page_source, 'html.parser')
    forwarding_notice = soup.find('div', class_='aXgaGb')
    if forwarding_notice:
        return True
    else:
        return False
