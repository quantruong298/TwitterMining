"""A simple sample program that accumulates tweets for the query \"#haiku\"
"""
import sys
import twitterpastcrawler

reload(sys)
sys.setdefaultencoding('utf8')

crawler = twitterpastcrawler.TwitterCrawler(query="", output_file="test.csv")

crawler.crawl()
