import os
import csv
from time import sleep
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from prefect import task, get_run_logger
from datetime import datetime
from config import SCRAPING_CONFIG, PATHS

@task
def scrape_medium_articles(tag: str, year: int):
    logger = get_run_logger()
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    stories_data = []
    os.makedirs(PATHS['raw_data'], exist_ok=True)
    csv_filename = os.path.join(PATHS['raw_data'], f"medium_{tag}_{year}.csv")
    
    columns = ['date', 'title', 'claps', 'responses', 'author_name', 'followers', 'reading_time_mins']

    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    url = f"{SCRAPING_CONFIG['base_url']}{tag}"
    logger.info(f"Starting scrape for tag: {tag}, year: {year}")

    for month in range(1, 13):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            n_days = 31
        elif month in [4, 6, 9, 11]:
            n_days = 30
        else:
            n_days = 29 if year % 4 == 0 else 28

        for day in range(1, n_days + 1):
            date_str = f"{month:02d}/{day:02d}/{year}"
            archive_url = f"{url}/archive/{year}/{month:02d}/{day:02d}"
            
            try:
                page = session.get(archive_url, timeout=10)
                soup = BeautifulSoup(page.text, 'html.parser')
                stories = soup.find_all('div', class_='streamItem streamItem--postPreview js-streamItem')
                
                if len(stories) > SCRAPING_CONFIG['max_stories_per_day']:
                    stories = stories[:SCRAPING_CONFIG['max_stories_per_day']]

                for story in stories:
                    try:
                        author_box = story.find('div', class_='postMetaInline u-floatLeft u-sm-maxWidthFullWidth')
                        if not author_box:
                            continue

                        author_link_tag = author_box.find('a')
                        author_url = author_link_tag['href']
                        author_name = author_url.split('@')[-1].strip('/')

                        reading_time = author_box.find('span', class_='readingTime')['title'].split()[0]
                        title = story.find('h3').text.strip() if story.find('h3') else '-'
                        title = title.replace('\n', ' ').replace('\t', ' ')

                        claps_tag = story.find('button', class_='button button--chromeless u-baseColor--buttonNormal js-multirecommendCountButton u-disablePointerEvents')
                        claps = claps_tag.text.strip().replace(',', '') if claps_tag else '0'

                        responses_tag = story.find('a', class_='button button--chromeless u-baseColor--buttonNormal')
                        responses = responses_tag.text.strip().split()[0] if responses_tag else '0'

                        story_data = [
                            date_str,
                            f'"{title}"' if ',' in title else title,
                            claps,
                            responses,
                            author_name,
                            'N/A',
                            reading_time
                        ]

                        stories_data.append(story_data)
                        if len(stories_data) >= 5:
                            with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerows(stories_data)
                            stories_data.clear()

                    except Exception as e:
                        logger.error(f"Error parsing story: {str(e)[:100]}")
                        continue

                sleep(SCRAPING_CONFIG['request_delay'])

            except Exception as e:
                logger.error(f"Failed to load {archive_url}: {str(e)[:100]}")
                continue

    if stories_data:
        with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(stories_data)

    logger.info(f"Completed scraping for tag: {tag}, year: {year}")
    return csv_filename