import os

from bs4 import BeautifulSoup
from profile_builder import interest_integrater
from selenium import webdriver
from utils import save_data

DATA_SAVE_DIR = 'data'


class DataBuilder:

    def crawl_euronews(self, topic: str):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")

        driver = webdriver.Chrome(options=chrome_options)
        driver.delete_all_cookies()

        url_head = "https://www.euronews.com/search?query=" + topic
        data = []
        n_pages = 100
        FIRST_PAGE = True

        for page in range(1, n_pages):
            url = url_head + "&p=" + str(page)
            driver.get(url)

            # handle 'Accept-Cookie' popup
            if FIRST_PAGE:
                button = driver.find_element_by_id("didomi-notice-agree-button")
                button.click()
                FIRST_PAGE = False

            # crawl title and short summary of the articles
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            object_bodies = soup.find_all('div', {"class": "m-object__body"})
            for each in object_bodies:
                title = each.find(class_="m-object__title__link")
                contents = each.find(class_="m-object__description")
                title = title.text.strip()
                if contents:
                    title += contents.text.strip()
                data.append(title)
        driver.quit()
        filepath = os.path.join(DATA_SAVE_DIR, topic)
        save_data(filepath, data)

    def build_training_data(self, new_topics=None):
        topics = interest_integrater.interests

        if new_topics:
            topics = [each for each in new_topics if each not in interest_integrater.interests]

        for each in topics:
            print("Crawling about {}... Please wait...".format(each))
            self.crawl_euronews(each)
