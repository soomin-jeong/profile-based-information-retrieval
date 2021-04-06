import os


from bs4 import BeautifulSoup
from profile_builder import interest_integrater
from selenium import webdriver


SAVE_DIR = 'data'


class DataBuilder:

    def crawl_euronews(self, topic:str):
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
        filepath = os.path.join(SAVE_DIR, topic)
        self.save_data(filepath, data)

    def save_data(self, filepath: str, data: list):
        try:
            f = open(filepath, 'w')
        except IOError:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            f = open(filepath, 'w')
        finally:
            for item in data:
                f.write("%s\n" % item)
            f.close()

    def build_training_data(self):
        for each in interest_integrater.interests:
            print("Crawling about", each)
            self.crawl_euronews(each)
