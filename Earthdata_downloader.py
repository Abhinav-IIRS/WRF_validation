# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:44:15 2023

@author: abhinavs
"""

from getpass import getpass
from pathlib import Path

from requests import Session
import platform

if platform.system() == "Windows":
    url_path = input("Enter path of the text file (Windows): ")
    url_path = url_path + '\\'
    
    dest = input("Enter destination folder (Windows): ")
    dest = dest + '\\'
    
else:
    url_path = input("Enter path of the text file (Linux): ")
    url_path = url_path + '/'
    
    dest = input("Enter destination folder (Linux): ")
    dest = dest + '/'
    
url_flnm = input("Enter text file name: ")

# This code will prompt you to enter your username and password
username = input("Earthdata username: ")
password = getpass("Earthdata password: ")



with open(url_path+url_flnm) as txtfile:
    for url in txtfile:
        # print(url)
        url = url[:-1]
        file_name = dest+Path(url).name
        session = Session()
        #session.auth = (username, password)
        _redirect = session.get(url)
        print("Downloading file: "+url)
        _response = session.get(_redirect.url, auth=(username, password))
        print("Download complete")
        print("-----------------")
        with open(file_name, 'wb') as file:
            file.write(_response._content)
