# https://qiita.com/Tanakadesu/items/5e00ec85fd746f225210
# Just get 2023 data (봉황전)

import os
import gzip
import requests
import time

folderpath = 'scraw2023/'
out_dir = 'data/'
for file in os.listdir(folderpath):
    if "202306" in file:
        break
    origin = folderpath + file
    date = out_dir+file[:-8]+".txt"
    f1 = gzip.open(origin,'rb')

    counter = 0
    while line := f1.readline():
        counter += 1
        url = line.decode('utf-8')[31:87]
        xml = url[:20]+url[21:24]+"/?"+url[25:]
        f2 = open(out_dir+xml[25:]+".xml", "w", encoding='utf-8')
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.90 Safari/537.36'}
        res=requests.get(url=xml, headers=headers)
        f2.write(res.text)
        f2.close()
    print(counter, " xml files found")
    f1.close()
    time.sleep(5)