import sys
import cloudscraper

def cloud_scrape():
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'linux'
        }
    )
    args = sys.argv[1:]

    print(scraper.get(args[0]).text)
       

if __name__ == "__main__":
    cloud_scrape()
    