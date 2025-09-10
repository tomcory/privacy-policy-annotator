import json
import re
from typing import Dict, Any, Optional

from bs4 import BeautifulSoup, Tag

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.state_manager import BaseStateManager


class PlayStoreParser(PipelineStep):
    """Parser for extracting app metadata from Google Play Store HTML snippets."""

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
            client: ApiBase = None
    ):
        super().__init__(
            run_id=run_id,
            task='parse_playstore',
            details='Extract app metadata from Google Play Store HTML',
            skip=skip,
            is_llm_step=False,
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

    async def execute(self, pkg: str):
        """Execute the PlayStore parsing step."""
        try:
            html_content = util.read_from_file(f"{self.in_folder}/{pkg}.html")
            if len(html_content) > 0:
                app_metadata = self._parse_playstore_html(html_content)
                util.append_to_file(f"{self.out_folder}/{pkg}.jsonl", json.dumps(app_metadata))
            else:
                await self.state_manager.raise_error(error_message=f"Empty HTML content for {pkg}")
        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))

    async def prepare_batch(self, pkg: str):
        """Prepare batch processing (not used for this parser)."""
        pass

    def _parse_playstore_html(self, html_content: str) -> Dict[str, Any]:
        """Parse Google Play Store HTML and extract app metadata."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Identify and extract relevant snippets
        appstats_snippet = self._extract_appstats_snippet(soup)
        appinfo_snippet = self._extract_appinfo_snippet(soup)
        devinfo_snippet = self._extract_devinfo_snippet(soup)
        
        # Parse each snippet for its respective metadata
        appstats_metadata = self._parse_appstats_snippet(appstats_snippet) if appstats_snippet else {}
        appinfo_metadata = self._parse_appinfo_snippet(appinfo_snippet) if appinfo_snippet else {}
        devinfo_metadata = self._parse_devinfo_snippet(devinfo_snippet) if devinfo_snippet else {}
        
        # Combine all metadata
        metadata = {**appstats_metadata, **appinfo_metadata, **devinfo_metadata}
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return metadata

    def _extract_appstats_snippet(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Extract the appstats snippet from the full HTML document."""
        appstats_div = soup.find('div', class_='dzkqwc')
        return appstats_div

    def _extract_appinfo_snippet(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Extract the appinfo snippet from the full HTML document."""
        # Find section with class "HcyOxe" containing a meta element with itemprop="description"
        sections = soup.find_all('section', class_='HcyOxe')
        for section in sections:
            meta_desc = section.find('meta', attrs={'itemprop': 'description'})
            if meta_desc:
                return section
        return None

    def _extract_devinfo_snippet(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Extract the devinfo snippet from the full HTML document."""
        # Find div with id "developer-contacts"
        devinfo_div = soup.find('div', id='developer-contacts')
        return devinfo_div

    def _parse_appstats_snippet(self, appstats_soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse the appstats snippet for app statistics metadata."""
        return {
            'app_name': self._extract_app_name(appstats_soup),
            'developer': self._extract_developer(appstats_soup),
            'rating': self._extract_rating(appstats_soup),
            'review_count': self._extract_review_count(appstats_soup),
            'download_count': self._extract_download_count(appstats_soup),
            'content_rating': self._extract_content_rating(appstats_soup),
            'features': self._extract_features(appstats_soup),
            'app_icon_url': self._extract_app_icon_url(appstats_soup),
            'package_id': self._extract_package_id(appstats_soup),
            'price': self._extract_price(appstats_soup),
            'install_button_text': self._extract_install_button_text(appstats_soup),
            'available_in_country': self._extract_available_in_country(appstats_soup)
        }

    def _parse_appinfo_snippet(self, appinfo_soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse the appinfo snippet for app information metadata."""
        return {
            'app_description': self._extract_app_description(appinfo_soup),
            'short_description': self._extract_short_description(appinfo_soup),
            'last_updated': self._extract_last_updated(appinfo_soup),
            'available_platforms': self._extract_available_platforms(appinfo_soup),
            'app_ranking': self._extract_app_ranking(appinfo_soup),
            'app_categories': self._extract_app_categories(appinfo_soup)
        }

    def _parse_devinfo_snippet(self, devinfo_soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse the devinfo snippet for developer information metadata."""
        return {
            'developer_website': self._extract_developer_website(devinfo_soup),
            'support_email': self._extract_support_email(devinfo_soup),
            'privacy_policy_url': self._extract_privacy_policy_url(devinfo_soup),
            'company_name': self._extract_company_name(devinfo_soup),
            'company_email': self._extract_company_email(devinfo_soup),
            'company_address': self._extract_company_address(devinfo_soup),
            'company_phone': self._extract_company_phone(devinfo_soup)
        }

    def _extract_app_name(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract app name from the HTML."""
        # Look for span with class "AfwdI" and itemprop="name"
        app_name_elem = soup.find('span', class_='AfwdI', attrs={'itemprop': 'name'})
        if app_name_elem:
            return app_name_elem.get_text(strip=True)
        
        # Fallback: look for h1 with app name
        h1_elem = soup.find('h1')
        if h1_elem:
            return h1_elem.get_text(strip=True)
        
        return None

    def _extract_developer(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract developer name from the HTML."""
        # Look for developer link in Vbfug class
        dev_elem = soup.find('div', class_='Vbfug')
        if dev_elem:
            link = dev_elem.find('a')
            if link:
                span = link.find('span')
                if span:
                    return span.get_text(strip=True)
        
        # Fallback: look for any link with developer pattern
        dev_links = soup.find_all('a', href=re.compile(r'/store/apps/dev\?id='))
        for link in dev_links:
            span = link.find('span')
            if span:
                return span.get_text(strip=True)
        
        return None

    def _extract_rating(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract app rating from the HTML."""
        # Look for rating in TT9eCd class
        rating_elem = soup.find('div', class_='TT9eCd')
        if rating_elem:
            rating_text = rating_elem.get_text(strip=True)
            # Extract numeric rating (e.g., "4.7" from "4.7★")
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                return float(rating_match.group(1))
        
        # Fallback: look for aria-label with rating
        rating_elems = soup.find_all(attrs={'aria-label': re.compile(r'Rated .* stars')})
        for elem in rating_elems:
            aria_label = elem.get('aria-label', '')
            rating_match = re.search(r'Rated (\d+\.?\d*) stars', aria_label)
            if rating_match:
                return float(rating_match.group(1))
        
        return None

    def _extract_review_count(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract review count from the HTML."""
        # Look for review count in g1rdde class
        review_elems = soup.find_all('div', class_='g1rdde')
        for elem in review_elems:
            text = elem.get_text(strip=True)
            if 'review' in text.lower():
                # Extract the review count (e.g., "2.2M reviews")
                review_match = re.search(r'([\d\.]+[KMB]?)\s*reviews?', text, re.IGNORECASE)
                if review_match:
                    return review_match.group(1)
        
        return None

    def _extract_download_count(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract download count from the HTML."""
        # Look for download count in ClM7O class followed by "Downloads"
        clm7o_elems = soup.find_all('div', class_='ClM7O')
        for elem in clm7o_elems:
            download_text = elem.get_text(strip=True)
            # Check if this element is followed by "Downloads"
            next_sibling = elem.find_next_sibling('div', class_='g1rdde')
            if next_sibling and 'download' in next_sibling.get_text(strip=True).lower():
                return download_text
        
        # Fallback: look for any text with download pattern
        download_elems = soup.find_all(text=re.compile(r'[\d\.]+[KMB]?\+\s*Downloads?', re.IGNORECASE))
        for elem in download_elems:
            download_match = re.search(r'([\d\.]+[KMB]?\+)', elem, re.IGNORECASE)
            if download_match:
                return download_match.group(1)
        
        return None

    def _extract_content_rating(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract content rating from the HTML."""
        # Look for content rating in itemprop="contentRating"
        rating_elem = soup.find(attrs={'itemprop': 'contentRating'})
        if rating_elem:
            span = rating_elem.find('span')
            if span:
                return span.get_text(strip=True)
        
        # Fallback: look for content rating text
        rating_elems = soup.find_all(text=re.compile(r'[A-Z]+:\s*[A-Za-z\s]+'))
        for elem in rating_elems:
            if ':' in elem and any(rating in elem for rating in ['USK:', 'PEGI:', 'ESRB:']):
                return elem.strip()
        
        return None

    def _extract_features(self, soup: BeautifulSoup) -> Optional[list]:
        """Extract app features from the HTML."""
        features = []
        
        # Look for features in UIuSk class
        feature_elems = soup.find_all('span', class_='UIuSk')
        for elem in feature_elems:
            feature_text = elem.get_text(strip=True)
            if feature_text:
                features.append(feature_text)
        
        return features if features else None

    def _extract_app_icon_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract app icon URL from the HTML."""
        # Look for app icon with itemprop="image"
        icon_elem = soup.find('img', attrs={'itemprop': 'image', 'alt': 'Icon image'})
        if icon_elem:
            return icon_elem.get('src')
        
        # Fallback: look for any image with icon-related classes
        icon_elems = soup.find_all('img', class_=re.compile(r'.*icon.*', re.IGNORECASE))
        for elem in icon_elems:
            src = elem.get('src')
            if src and 'play-lh.googleusercontent.com' in src:
                return src
        
        return None

    def _extract_package_id(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract package ID from the HTML."""
        # Look for package ID in data-item-id attribute
        item_elems = soup.find_all(attrs={'data-item-id': True})
        for elem in item_elems:
            item_id = elem.get('data-item-id', '')
            # Extract package ID from data-item-id (e.g., "com.block.juggle")
            package_match = re.search(r'"([^"]+)"', item_id)
            if package_match:
                return package_match.group(1)
        
        # Fallback: look for package ID in any data attribute
        data_elems = soup.find_all(attrs=re.compile(r'data-.*'))
        for elem in data_elems:
            for attr_name, attr_value in elem.attrs.items():
                if attr_name.startswith('data-') and isinstance(attr_value, str):
                    package_match = re.search(r'([a-z]+\.[a-z]+\.[a-z]+)', attr_value)
                    if package_match:
                        return package_match.group(1)
        
        return None

    def _extract_price(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract app price from the HTML."""
        # Look for price in meta tag with itemprop="price"
        price_elem = soup.find('meta', attrs={'itemprop': 'price'})
        if price_elem:
            price = price_elem.get('content')
            if price == '0':
                return 'Free'
            return price
        
        # Fallback: look for price text
        price_elems = soup.find_all(text=re.compile(r'\$[\d\.]+|Free|Paid', re.IGNORECASE))
        for elem in price_elems:
            price_text = elem.strip()
            if price_text:
                return price_text
        
        return None

    def _extract_install_button_text(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract install button text from the HTML."""
        # Look for install button
        install_buttons = soup.find_all('button', attrs={'aria-label': 'Install'})
        for button in install_buttons:
            text_elem = button.find('span', class_='VfPpkd-vQzf8d')
            if text_elem:
                return text_elem.get_text(strip=True)
        
        # Fallback: look for any button with install-related text
        buttons = soup.find_all('button')
        for button in buttons:
            button_text = button.get_text(strip=True)
            if 'install' in button_text.lower():
                return button_text
        
        return None

    def _extract_developer_website(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract developer website from the HTML."""
        # Look for website link with "Website" text
        website_links = soup.find_all('a', href=True)
        for link in website_links:
            website_text = link.find('div', class_='xFVDSb')
            if website_text and 'Website' in website_text.get_text():
                href = link.get('href')
                if href and 'http' in href:
                    return href
        return None

    def _extract_support_email(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract support email from the HTML."""
        # Look for support email in pSEeg class
        support_email_elem = soup.find('div', class_='pSEeg')
        if support_email_elem:
            email_text = support_email_elem.get_text(strip=True)
            if '@' in email_text:
                return email_text
        
        # Fallback: look for mailto links
        mailto_links = soup.find_all('a', href=re.compile(r'^mailto:'))
        for link in mailto_links:
            href = link.get('href')
            if href and 'mailto:' in href:
                email = href.replace('mailto:', '')
                if '@' in email:
                    return email
        return None

    def _extract_privacy_policy_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract privacy policy URL from the HTML."""
        # Look for privacy policy link
        privacy_links = soup.find_all('a', href=True)
        for link in privacy_links:
            privacy_text = link.find('div', class_='xFVDSb')
            if privacy_text and 'Privacy Policy' in privacy_text.get_text():
                href = link.get('href')
                if href and 'http' in href:
                    return href
        return None

    def _extract_company_name(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract company name from the HTML."""
        # Look for company name in HhKIQc class
        company_info = soup.find('div', class_='HhKIQc')
        if company_info:
            divs = company_info.find_all('div')
            if divs:
                company_name = divs[0].get_text(strip=True)
                if company_name and not '@' in company_name and not '+' in company_name:
                    return company_name
        return None

    def _extract_company_email(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract company email from the HTML."""
        # Look for company email in HhKIQc class
        company_info = soup.find('div', class_='HhKIQc')
        if company_info:
            divs = company_info.find_all('div')
            for div in divs:
                email_text = div.get_text(strip=True)
                if '@' in email_text and 'hungrystudio.com' in email_text:
                    return email_text
        return None

    def _extract_company_address(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract company address from the HTML."""
        # Look for company address in mHsyY class
        address_elem = soup.find('div', class_='mHsyY')
        if address_elem:
            address_text = address_elem.get_text(strip=True)
            if address_text and len(address_text) > 10:
                # Clean up the address formatting
                address_lines = [line.strip() for line in address_text.split('\n') if line.strip()]
                return ', '.join(address_lines)
        
        # Fallback: look for address-like text in HhKIQc
        company_info = soup.find('div', class_='HhKIQc')
        if company_info:
            divs = company_info.find_all('div')
            for div in divs:
                text = div.get_text(strip=True)
                if re.search(r'\d+.*[A-Z\s]+', text) and len(text) > 15:
                    return text
        return None

    def _extract_company_phone(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract company phone from the HTML."""
        # Look for phone number in HhKIQc class
        company_info = soup.find('div', class_='HhKIQc')
        if company_info:
            divs = company_info.find_all('div')
            for div in divs:
                phone_text = div.get_text(strip=True)
                if re.search(r'^\+?\d[\d\s\-\(\)]+$', phone_text):
                    return phone_text
        return None

    def _extract_app_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract app description from the HTML."""
        # Look for app description in bARER class
        description_elem = soup.find('div', class_='bARER')
        if description_elem:
            description_text = description_elem.get_text(strip=True)
            if description_text:
                # Clean up the description formatting
                # Replace multiple whitespace with single space
                description_text = re.sub(r'\s+', ' ', description_text)
                # Remove extra newlines and formatting
                description_text = description_text.replace('\n', ' ').strip()
                return description_text
        return None

    def _extract_short_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract short description from the HTML."""
        # Look for short description in meta tag with itemprop="description"
        description_elem = soup.find('meta', attrs={'itemprop': 'description'})
        if description_elem:
            description_text = description_elem.get('content')
            if description_text:
                return description_text
        return None

    def _extract_last_updated(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract last updated date from the HTML."""
        # Look for last updated date in TKjAsc section
        tkj_asc = soup.find('div', class_='TKjAsc')
        if tkj_asc:
            # Look for "Updated on" label
            updated_label = tkj_asc.find('div', class_='lXlx5', string='Updated on')
            if updated_label:
                # Get the next sibling div with the date
                next_div = updated_label.find_next_sibling('div', class_='xg1aie')
                if next_div:
                    return next_div.get_text(strip=True)
        return None

    def _extract_available_platforms(self, soup: BeautifulSoup) -> Optional[list]:
        """Extract available platforms from the HTML."""
        platforms = []
        
        # Look for available platforms in TKjAsc section
        tkj_asc = soup.find('div', class_='TKjAsc')
        if tkj_asc:
            # Look for "Available on" label
            available_label = tkj_asc.find('div', class_='lXlx5', string='Available on')
            if available_label:
                # Get the next sibling div with the platforms
                next_div = available_label.find_next_sibling('div', class_='xg1aie')
                if next_div:
                    platform_text = next_div.get_text(strip=True)
                    # Split by comma and clean up
                    platforms = [p.strip() for p in platform_text.split(',') if p.strip()]
                    return platforms
        return None

    def _extract_app_ranking(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract app ranking from the HTML."""
        # Look for app ranking in button text within Uc6QCc section
        ranking_buttons = soup.find_all('button', class_='VfPpkd-LgbsSe')
        for button in ranking_buttons:
            ranking_text = button.get_text(strip=True)
            # Look for ranking patterns like "#3 top for €0 puzzle"
            if re.search(r'#\d+.*top.*', ranking_text):
                return ranking_text
        return None

    def _extract_app_categories(self, soup: BeautifulSoup) -> Optional[list]:
        """Extract app categories from the HTML."""
        categories = []
        
        # Look for categories in Uc6QCc section
        uc6_qcc = soup.find('div', class_='Uc6QCc')
        if uc6_qcc:
            # Look for all category spans
            category_spans = uc6_qcc.find_all('span', class_='VfPpkd-vQzf8d')
            for span in category_spans:
                category_text = span.get_text(strip=True)
                # Filter out ranking text and empty strings
                if category_text and not re.search(r'#\d+.*top.*', category_text):
                    categories.append(category_text)
        
        return categories if categories else None

    def _extract_available_in_country(self, soup: BeautifulSoup) -> Optional[bool]:
        """Check if the app is available in the country by looking for an Install button in the appstats snippet."""
        # The appstats snippet is already the div with class 'dzkqwc'
        install_button = soup.find('button', attrs={'aria-label': 'Install'})
        if install_button:
            return True
        else:
            return False


def parse_playstore_html(html_content: str) -> Dict[str, Any]:
    """
    Standalone function to parse Google Play Store HTML and extract app metadata.
    
    Args:
        html_content (str): HTML content from a Google Play Store page
        
    Returns:
        Dict[str, Any]: Dictionary containing extracted app metadata
    """
    parser = PlayStoreParser(
        run_id='standalone',
        skip=False,
        in_folder='.',
        out_folder='.',
        state_manager=None
    )
    return parser._parse_playstore_html(html_content)
