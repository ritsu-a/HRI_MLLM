#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2024-04-24
@author: wangzhongbin
'''

import os
import time
import hashlib
import requests
from .protocol import Protocol
from tts.common.logger import LOGGER,ExecutionTime

class Fun(Protocol):
    '''
    FunASR 音频转文字类
    '''
    
    def __init__(self):
        super().__init__()
        SECRET_KEY = os.getenv("ASR_SECRET")
        self.timestamp = str(int(time.time()))

        data = f"{SECRET_KEY}{self.timestamp}"
        md5_hash = hashlib.md5()
        md5_hash.update(data.encode('utf-8'))
        self.token = md5_hash.hexdigest()
        self.uri = os.getenv("ASR_URI")

    @ExecutionTime("")
    def to_text(self, audio, lang = 'auto', code='mp3',ratio='16k'):
        '''
        音频转文字
        @param audio: 音频数据
        @param lang: 语种en英 zh中 mix中英混合
        @param code: 编码 mp3, pcm
        @param ratio: 码率 16k
        @return string: 返回文字，失败返回 None
        '''
        try:
            url = f"{self.uri}/?token={self.token}&time={self.timestamp}&language={lang}"
            response = requests.post(url, data=audio,timeout=5)
            res = response.json()
            return res['text'] if res['text'] is not None else None
        except Exception as e:
            LOGGER.error(f"❗ 发送时发生错误: {e}")
        return None