import traceback
import requests
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from youtube_converter import YoutubeConverter

os.chdir('./assets/Tracks_and_Covers')

url = 'https://music.youtube.com/watch?v=v425V7WckYc&list=RDAMVMdfSr0b0eFmo'
browser = webdriver.Chrome()
browser.get(url)




while True:

    url = browser.current_url

    track_name = browser.find_element(By.XPATH, '/html/body/ytmusic-app/ytmusic-app-layout/ytmusic-player-bar/div[2]/div[2]/yt-formatted-string')
    title = track_name.get_attribute('title')
    print(title)

    os.mkdir(title)
    os.chdir(title)

    img = browser.find_element(By.XPATH,'//*[@id="img"]')
    src = img.get_attribute('src')
    print(src)

    response = requests.get(src)
    open("cover.png", "wb").write(response.content)
    # YoutubeConverter.convert(url)
    os.chdir('../')

    next = browser.find_element(By.XPATH, '/html/body/ytmusic-app/ytmusic-app-layout/ytmusic-player-bar/div[1]/div/tp-yt-paper-icon-button[5]')
    next.click()