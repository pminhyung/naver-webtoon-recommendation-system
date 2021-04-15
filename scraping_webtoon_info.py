import os
import argparse
import requests
import ssl
ssl._create_default_https_context= ssl._create_unverified_context
from urllib.request import urlretrieve

from bs4 import BeautifulSoup
import openpyxl


# Save info of webtoons_still on (sort by day of week)
def save_stillon_info(save_dir = None):
    """
    scrap and save metadata of webtoons in series (title, artist, genre)
    :return:
    """
    wb1 = openpyxl.Workbook()
    sheet1 = wb1.active

    url = 'https://comic.naver.com/webtoon/weekday.nhn'
    raw = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

    html = BeautifulSoup(raw.text, 'html.parser')

    sheet1.append(['제목', '작가', '장르'])

    webtoons = html.select('div.list_area li')

    for i in webtoons:
        front_url = 'https://comic.naver.com'
        each_url = front_url + i.select_one('a').attrs['href']
        print(each_url)

        raw_each = requests.get(each_url, headers={'User-Agent': 'Mozilla/5.0'})
        html_each = BeautifulSoup(raw_each.text, 'html.parser')
        head = html_each.select_one('div.detail').select_one('h2').text.replace(' ', '').strip()

        title = head.split()[0].replace('휴재', '')
        artist = head.split()[1]
        genre = html_each.select_one('div.detail').select_one('span.genre').text
        print(title)
        print(artist)
        print(genre)
        print('=' * 20)
        sheet1.append([title, artist, genre])

    sheet1.title = 'NaverWebtoon'

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fname = save_dir + 'NaverWebtoon_stillon.xlsx'
    else:
        fname = 'NaverWebtoon_stillon.xlsx'
    wb1.save(fname)

# Save info of webtoons_finished (sort by views)
def save_finished_info():
    """
    scrap and save metadata of completed webtoons(title, artist, genre)
    :return:
    """
    wb2 = openpyxl.Workbook()
    sheet2 = wb2.active

    url_fin = 'https://comic.naver.com/webtoon/finish.nhn?order=ViewCount&view=image'
    raw_fin = requests.get(url_fin, headers={'User-Agent': 'Mozilla/5.0'})

    html_fin = BeautifulSoup(raw_fin.text, 'html.parser')

    sheet2.append(['제목', '작가', '장르'])

    finished_webtoons = html_fin.select('div.list_area li')

    for i in finished_webtoons[:55]:
        front_url = 'https://comic.naver.com'
        each_url = front_url + i.select_one('a').attrs['href']
        print(each_url)

        raw_each = requests.get(each_url, headers={'User-Agent': 'Mozilla/5.0'})
        html_each = BeautifulSoup(raw_each.text, 'html.parser')
        head = html_each.select_one('div.detail').select_one('h2').text.replace(' ', '').strip()

        title = head.split()[0].replace('휴재', '')
        artist = head.split()[1]
        genre = html_each.select_one('div.detail').select_one('span.genre').text
        print(title)
        print(artist)
        print(genre)
        print('=' * 20)
        sheet2.append([title, artist, genre])

    sheet2.title = 'NaverWebtoon_finished'

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fname = save_dir + 'NaverWebtoon_finished.xlsx'
    else:
        fname = 'NaverWebtoon_finished.xlsx'
    wb1.save(fname)

# Save Thumbnails
def save_thumbnails(save_dir=None):
    """
    scrap and save thubnails of webtoons

    :param save_directory: directory to save (default - current working directory)
                        ex) "C:/Users/user/PycharmProjects/imgsave/image"
    :return: None
    """
    raw_thumbnail= requests.get("https://comic.naver.com/webtoon/weekday.nhn",
                                headers={'User-Agent':'Mozilla/5.0'})
    html_thumbnail= BeautifulSoup(raw_thumbnail.text,'html.parser')
    webtoon = html_thumbnail.select("div.list_area.daily_all  div ul li")

    if save_dir:
        if not os.path.exists(directory):
            os.makedirs(directory)
        dir_name = save_dir + '/'
    else:
        dir_name = ''
    for w in webtoon :
        title= w.select_one("div.thumb a img").attrs["title"]
        img= w.select_one("div.thumb a img").attrs["src"]
        urlretrieve(img, dir_name+str(title)+".jpg")
        print(title, "Thumbnails saved!")


def main(data_dir, img_dir):
    save_stillon_info(save_dir=data_dir)
    save_finished_info(save_dir=data_dir)
    save_thumbnails(save_dir=img_dir)

if __name__ == '__main__':

    # scrap and save metadata to make survey form
    parser = argparse.ArgumentParser(description='scraping parser')
    parser.add_argument('-d', '--data_dir', help='config_file')
    parser.add_argument('-i', '--img_dir', help='labels', action='append')
    args = parser.parse_args()

    if args.data_dir and args.img_dir:
        main(args.data_dir, args.img_dir)
    else:
        data_dir = "C:/Users/user/PycharmProjects/webtoon_info"
        img_dir = "C:/Users/user/PycharmProjects/imgsave/image"
        main(data_dir, img_dir)