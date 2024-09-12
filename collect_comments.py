from selenium import webdriver as wd
from bs4 import BeautifulSoup
import time
import pandas as pd
import requests
import re
from datetime import datetime
import os
import glob

def get_urls_from_trend(keyword):
    titles = []
    urls = []
    infos =[]
    url = "https://www.youtube.com/feed/trending?bp=6gQJRkVleHBsb3Jl"
    
    driver = wd.Chrome()
    driver.get(url)

    last_page_height = driver.execute_script("return document.documentElement.scrollHeight")
    idx = 0
    while True:
        idx += 1
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(3.0)
        
        new_page_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_page_height == last_page_height:
            break
        last_page_height = new_page_height

    html_source = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(html_source, 'lxml')
    datas = soup.select("a#video-title")
    for data in datas:
        title = data.text.replace('\n', '')
        url = "https://www.youtube.com/" + data.get('href')
        info = data.get('aria-label')
        
        titles.append(title)
        urls.append(url)
        infos.append(info)
    file_name = keyword + '_' + datetime.today().strftime("%Y%m%d%H%M%S")
        
    return titles, urls, file_name

def crawl_youtube_page_html_sources(urls):
    html_sources = []
    alluser = 0
    i = 0
    
    while alluser < 50000:
        driver = wd.Chrome()
        driver.get(urls[i])

        last_page_height = driver.execute_script("return document.documentElement.scrollHeight")

        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(1.0)
            new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(1.0)
            new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(1.0)
            new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

            if new_page_height == last_page_height:
                break
            last_page_height = new_page_height


        html_source = driver.page_source
        
        soup = BeautifulSoup(html_source, 'lxml')
        youtube_user_IDs = soup.select('div#header-author a#author-text')
        alluser += len(youtube_user_IDs)
        html_sources.append(html_source)
        print("OK")
        driver.quit()
        i += 1
        print(alluser)
    return html_sources

def get_user_IDs_and_comments(html_sources):
    my_dataframes = []
    for html in html_sources:
        soup = BeautifulSoup(html, 'lxml')
        
        youtube_user_IDs = soup.select('div#header-author a#author-text')
        youtube_comments = soup.select('yt-formatted-string#content-text')
        print(len(youtube_user_IDs))
        str_youtube_userIDs = []
        str_youtube_comments = []
        
        for i in range(len(youtube_user_IDs)):
            try:
                str_tmp1 = str(youtube_user_IDs[i].text)
                str_tmp1 = str_tmp1.replace('\n', '')
                str_tmp1 = str_tmp1.replace('\t', '')
                str_tmp1 = str_tmp1.replace('                ','')

                str_tmp2 = str(youtube_comments[i].text) 
                str_tmp2 = str_tmp2.replace('\n', '')
                str_tmp2 = str_tmp2.replace('\t', '')
                str_tmp2 = str_tmp2.replace('               ', '')

                str_youtube_comments.append(str_tmp2)
                str_youtube_userIDs.append(str_tmp1)
            except:
                pass
            
        pd_data = {"ID":str_youtube_userIDs, "Comment":str_youtube_comments}
        youtube_pd = pd.DataFrame(pd_data)

        my_dataframes.append(youtube_pd)
        
    return my_dataframes

# 영상별 댓글을 저장한 폴더를 생성 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
# 영상별 댓글을 영상제목.csv 파일로 저장. 위에서 만든 폴더에        
def convert_csv_from_dataframe(file_name ,titles, my_dataframes):
    for i in range(len(my_dataframes)):
        title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', titles[i])
        
        folder_path = "./" + file_name
        createFolder(folder_path)
        
        save_path = "./" + file_name + '/'
        my_dataframes[i].to_csv("{}{}.csv".format(save_path, title), index=False, encoding='utf-8-sig')
        
# 영상제목.csv 파일들을 1개의 csv 파일로 만듬 
def make_one_csv_from_folder(file_name):
    
    folder_path = "./" + file_name
    all_data = []
    for f in glob.glob( folder_path + "/*.csv"):
           all_data.append(pd.read_csv(f))

    df = pd.concat(all_data, ignore_index=True)
    
    df.to_csv(file_name + '_commnet_total'+ '.csv', index = False, encoding='utf-8-sig')

titles, urls, file_name = get_urls_from_trend(keyword)
html_sorces = crawl_youtube_page_html_sources(urls)
my_dataframes = get_user_IDs_and_comments(html_sorces)
convert_csv_from_dataframe(file_name, titles, my_dataframes)
make_one_csv_from_folder(file_name)
