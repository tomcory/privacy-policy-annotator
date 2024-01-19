import re
import time

from bs4 import BeautifulSoup, NavigableString, Comment
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import util


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


def crawl_list(pkgs: list[str], retries: int = 2):
    for pkg_name in pkgs:

        policy = fetch_policy(pkg_name, retries)

        if len(policy) > 0:
            print('Saving original HTML to file...')
            file_name = 'original/' + pkg_name + '.html'
            util.write_to_file(file_name, policy)
        else:
            print('Empty original policy, not saving to file')
            continue

        policy = clean_policy(policy)

        if len(policy) > 0:
            print('Saving cleaned HTML to file...')
            file_name = 'cleaned/' + pkg_name + '.html'
            util.write_to_file(file_name, policy)
        else:
            print('Empty cleaned policy, not saving to file')


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


def clean_policy(page) -> str:
    try:
        print('Attempting bs4 extract...')
        policy_bs4 = extract_policy_from_page_bs4(page)
        if policy_bs4:
            print('bs4 extract successful.')
            policy = policy_bs4
        else:
            print('bs4 extract failed')
            raise Exception

        print('Prettifying soup...')

        cleaned_policy = policy.prettify()
        return cleaned_policy

    except Exception as e:
        print('Error: Unknown exception %s' % e)
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


def extract_policy_from_page_bs4(page_source) -> BeautifulSoup:
    """
    Extract the policy text from the page source using BeautifulSoup.

    Args:
        page_source (str): The HTML source of the page.

    Returns:
        BeautifulSoup: The isolated and cleaned main content of the page source.
    """
    try:
        soup = BeautifulSoup(page_source, 'html.parser')

        # reduce the soup to just the body
        body = soup.find('body')
        if body is not None:
            print('Reducing soup to body...')
            body_soup = BeautifulSoup(str(body), 'html.parser')
            soup = body_soup
        else:
            print('No body found?!')
            raise Exception

        # if there's a main element, reduce the soup to that
        main = soup.find("main")
        if main is not None:
            print('Main (tag) found, reducing soup...')
        else:
            # some sites use the role 'article' instead of the tag itself
            article = soup.find("div", {"role": re.compile('.*[mM]main.*')})
            if main is not None:
                print('Main (role) found, reducing soup...')
            else:
                # some sites use the class 'article' instead of the tag itself
                main = soup.find("div", {"role": re.compile('.*[mM]main.*')})
                if main is not None:
                    print('Main (role) found, reducing soup...')
                else:
                    # and some sites use the id 'article' instead of the tag itself
                    main = soup.find("div", id=re.compile('.*[mM]main.*'))
                    if main is not None:
                        print('Main (role) found, reducing soup...')
        if main is not None:
            # remove all elements of the soup except the article and its children
            new_soup = BeautifulSoup(str(main), 'html.parser')
            # Replace the old soup with the new one
            soup = new_soup

        # if there's an article element, reduce the soup to that
        article = soup.find("article")
        if article is not None:
            print('Article (tag) found, reducing soup...')
        else:
            # some sites use the role 'article' instead of the tag itself
            article = soup.find("div", {"role": re.compile('.*[aA]rticle.*')})
            if article is not None:
                print('Article (role) found, reducing soup...')
            else:
                # some sites use the class 'article' instead of the tag itself
                article = soup.find("div", {"role": re.compile('.*[aA]rticle.*')})
                if article is not None:
                    print('Article (role) found, reducing soup...')
                else:
                    # and some sites use the id 'article' instead of the tag itself
                    article = soup.find("div", id=re.compile('.*[aA]rticle.*'))
                    if article is not None:
                        print('Article (role) found, reducing soup...')
        if article is not None:
            # remove all elements of the soup except the article and its children
            new_soup = BeautifulSoup(str(article), 'html.parser')
            # Replace the old soup with the new one
            soup = new_soup

        # remove all script elements
        for script in soup.find_all("script"):
            print('Decomposing script...')
            script.decompose()

        # remove all style elements
        for style in soup.find_all("style"):
            print('Decomposing style...')
            style.decompose()

        # remove all header elements
        headers = soup.find_all('header')
        for header in headers:
            print('Decomposing header...')
            header.decompose()

        # remove all nav elements
        navs = soup.find_all('nav')
        for nav in navs:
            print('Decomposing nav...')
            nav.decompose()

        # remove all footer elements
        footers = soup.find_all('footer')
        for footer in footers:
            print('Decomposing footer...')
            footer.decompose()

        # remove all comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

        # remove all tags that have classnames or ids related to nav, header, and footer
        search_strings = ['.*nav.*', '.*header.*', '.*footer.*']
        for s in search_strings:
            regex = re.compile(s)
            for match_class in soup.find_all("div", {"class": regex}):
                print('Decomposing (class) %s' % match_class['class'])
                # match_class.decompose()
            for match_role in soup.find_all("div", {"role": regex}):
                print('Decomposing (role) %s' % match_role['role'])
                # match_role.decompose()
            for match_id in soup.find_all('div', id=regex):
                print('Decomposing (id) %s' % match_id['id'])
                # match_id.decompose()

        # Extended list of inline element tags to unwrap
        inline_tags = ['a', 'abbr', 'acronym', 'b', 'bdi', 'bdo', 'big', 'button',
                       'cite', 'code', 'data', 'del', 'dfn', 'em', 'i', 'img', 'input',
                       'ins', 'kbd', 'label', 'map', 'mark', 'meter', 'noscript', 'object',
                       'output', 'picture', 'progress', 'q', 'ruby', 's', 'samp', 'script',
                       'select', 'small', 'span', 'strong', 'sub', 'sup', 'svg', 'template',
                       'textarea', 'time', 'tt', 'u', 'var', 'wbr']

        highlight_tags = ['b', 'strong', 'i', 'em', 'big', 'u']

        headline_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

        # Find all headline tags (h1 through h6)
        headline_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        print('Finding lowest-order headline tag, there are %s headlines' % headline_tags)

        # Determine the highest-order headline tag
        least_significant_headline = min(5, max((int(tag.name[1]) for tag in headline_tags), default=1))

        # unwrap all inline tags as defined by the list above
        for tag in inline_tags:
            for inline_element in soup.find_all(tag):
                if len(inline_element.parent.contents) == 1 and inline_element.name in highlight_tags and \
                        inline_element.parent.name not in headline_tags:
                    replacement_tag = 'h' + str(least_significant_headline + 1)
                    print('Replacing standalone %s with %s: %s' % (
                    inline_element.name, replacement_tag, inline_element.text))
                    h4_element = soup.new_tag(replacement_tag)
                    h4_element.string = inline_element.text
                    inline_element.replace_with(h4_element)
                else:
                    inline_element.unwrap()

        print('Inline tags unwrapped.')
        print('Flattening soup...')

        # flatten the soup, unwrapping all elements that don't have text directly within them
        allowed_wrapper_tags = ['ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'td', 'th', 'br']
        tags = [tag for tag in soup.find_all() if tag.name is not None]
        for tag in tags:
            # Check if tag has no text of its own
            if (not any(isinstance(child, NavigableString) and child.strip() for child in tag.children)) and (
                    tag.name not in allowed_wrapper_tags):
                tag.unwrap()

        print('Concatenating text...')

        # remove all attributes from each tag and concatenate text
        for tag in soup.find_all():  # True finds all tags
            tag.attrs = {}

            # text can be split into multiple content entries; concatenate adjacent entries
            i = 0
            while i < len(tag.contents):
                content = tag.contents[i]

                # if the content is a NavigableString, start concatenation
                if isinstance(content, NavigableString):
                    concatenated_text = content

                    # continue concatenating adjacent NavigableStrings
                    while i + 1 < len(tag.contents) and (isinstance(tag.contents[i + 1], NavigableString) or
                                                         tag.contents[i + 1].name == 'br'):
                        if tag.contents[i + 1].name == 'br':
                            print('Replacing <br>')
                            concatenated_text += '\n '
                        else:
                            concatenated_text += tag.contents[i + 1]
                        del tag.contents[i + 1]  # Remove the next item as it's concatenated

                    # normalize whitespace in the concatenated text
                    # concatenated_text = ' '.join(concatenated_text.split())

                    # replace the original string with the concatenated, normalized one
                    tag.contents[i].replace_with(concatenated_text.replace(' ', ' '))

                i += 1  # Move to the next content item

        print('Soup cleaned.')

        return soup
    except Exception as e:
        print(
            f'An error occurred while extracting policy from text via bs4: {type(e)}. Fallback to page source extractor')
        return None


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
