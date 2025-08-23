#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-03-20
@author: wangzhongbin
'''

import os
import hmac
import json
import time
import base64
import hashlib
import datetime
import asyncio
import websockets
from time import mktime
from datetime import datetime
from tts.common.logger import LOGGER,ExecutionError,ExecutionTime
from .protocol import Protocol
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

class XfYunG1(Protocol):
    '''
    XfYun 文字转音频类
    '''

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("TTS_XFYUN_KEY")
        self.voice = os.getenv("TTS_XFYUN_VOICE")
        self.api_secret = os.getenv("TTS_XFYUN_SECRET")
        self.common_args = {"app_id": os.getenv("TTS_XFYUN_APPID")}
        self.business_args = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": self.voice, "tte": "utf8", "speed": 40}
        self.handler.rate = 16000

    def __create_url(self):
        '''
        生成url
        '''
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        date = format_date_time(mktime(datetime.now().timetuple()))
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        signature_sha = hmac.new(self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.api_key, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        url = url + '?' + urlencode(v)
        return url
    
    @ExecutionError
    def text_to_audio_stream(self, text:str, filepath:str, async_play:bool):
        '''
        文字转音频（同步版本）
        @param text: 文字
        @return BytesIO: 返回音频
        '''
        try:
            loop = asyncio.get_running_loop()
            fun = self.__text_to_audio_stream(text, filepath, async_play)
            if loop.is_running():
                return loop.create_task(fun)
                # return asyncio.run_coroutine_threadsafe(fun, loop).result()
            else:
                return loop.run_until_complete(fun)
        except RuntimeError as e:
            LOGGER.warning(e)
            return asyncio.run(self.__text_to_audio_stream(text, filepath, async_play))
    
    async def __text_to_audio_stream(self, text:str, filepath:str, async_play:bool):
        '''
        文字转音频（内部异步方法）
        '''
        self.business_args["auf"]=f"audio/L16;rate={self.handler.rate}"
        data = {
            "common": self.common_args,
            "business": self.business_args,
            "data": {"status": 2, "text": str(base64.b64encode(text.encode('utf-8')), "UTF8")},
        }
        request = json.dumps(data)
        byte_data = b""
        async with websockets.connect(self.__create_url()) as ws:
            await ws.send(request)
            while True:
                msg = await ws.recv()
                done = self.__response(msg, filepath, async_play, byte_data)
                if done:
                    break
            LOGGER.info("closing the connection...")
        return byte_data

    def __response(self, msg, filepath, async_play, byte_data):
        try:
            message =json.loads(msg)
            code = message["code"]
            if code != 0:
                LOGGER.error("sid:%s call error:%s code is:%s" % (sid, message["message"], code))
                return True
            sid = message["sid"]
            audio = message["data"]["audio"]
            status = message["data"]["status"]
            if audio is not None:
                audio = base64.b64decode(audio)
                if async_play:
                    super().play_by_udp(audio)
                    self.handler.play_audio(audio)
                if self.start_time is not None:
                    end_time = time.time()
                    LOGGER.info(f"TTS XfYun 首帧耗时: {end_time - self.start_time:.4f} 秒")
                    self.start_time = None
                with open(filepath, 'ab') as f:
                    f.write(audio)
                byte_data += audio
            return (status == 2)
        except Exception as e:
            LOGGER.error("receive msg,but parse exception:%s" % (e))
            return True

    def to_audio(self, text:str, async_play:bool = False):
        '''
        文字转音频
        @param str: 文字
        @param async_play: 是否异步播放
        @return bytes: 返回音频
        '''
        if text is None or isinstance(text,str) == False or text == '':
            msg = 'text 是空的'
            LOGGER.warning(msg)
            return None
        
        filename = 'xfyun_'+self.md5(self.voice+text)+'.pcm'
        filepath = os.path.join(self.wav_dir,filename)
        
        audio_stream = None
        if os.path.exists(filepath) :
            with open(filepath, "rb") as f:
                audio_stream = f.read()  # 读取文件数据
            if async_play:
                super().play_by_udp(audio_stream)  
                self.handler.play_audio(audio_stream)
        else :
            self.start_time = time.time()
            audio_stream = self.text_to_audio_stream(text, filepath, async_play)
        return audio_stream

    def play(self, text:str):
        '''
        播放音频
        @param text: 文字
        '''
        self.to_audio(text, True)
