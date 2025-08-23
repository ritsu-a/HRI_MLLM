#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-03-22
@author: wangzhongbin
'''

import os
import uuid
import json
import gzip
import copy
import time
import asyncio
import websockets
from .protocol import Protocol
from tts.common.logger import LOGGER,ExecutionError

class Volcengine(Protocol):
    '''
    Volcengine 文字转音频类
    '''
    
    def __init__(self):
        super().__init__()
        self.handler.rate = 24000
        self.token = os.getenv("TTS_VOLCENGINE_TOKEN")
        self.voice = os.getenv("TTS_VOLCENGINE_VOICE")
        self.api_url = f"wss://openspeech.bytedance.com/api/v1/tts/ws_binary"
        self.default_header = bytearray(b'\x11\x10\x11\x00')
        self.request_data = {
            "app": {
                "appid": os.getenv("TTS_VOLCENGINE_APPID"),
                "token": "access_token",
                "cluster": os.getenv("TTS_VOLCENGINE_CLUSTER")
            },
            "user": {
                "uid": "37"
            },
            "audio": {
                "voice_type": self.voice,
                "encoding": "pcm",
                "rate": self.handler.rate,
                "speed_ratio": 1,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": "",
                "text_type": "plain",
                "operation": "submit"
            }
        }

    @ExecutionError
    def text_to_audio_stream(self, text:str, filepath:str, async_play:bool):
        '''
        文字转音频（同步版本）
        @param text: 文字
        @return bytes: 返回音频
        '''
        try:
            loop = asyncio.get_running_loop()
            fun = self.__text_to_audio_stream(text, filepath, async_play)
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(fun, loop).result()
            else:
                return loop.run_until_complete(fun)
        except RuntimeError as e:
            LOGGER.warning(e)
            return asyncio.run(self.__text_to_audio_stream(text, filepath, async_play))
    
    async def __text_to_audio_stream(self, text:str, filepath:str, async_play:bool):
        '''
        文字转音频（内部异步方法）
        '''
        self.request_data["audio"]["rate"]=self.handler.rate
        request = copy.deepcopy(self.request_data)
        request["request"]["reqid"] = str(uuid.uuid4())
        request["request"]["text"] = text
        payload_bytes = str.encode(json.dumps(request))
        payload_bytes = gzip.compress(payload_bytes) 
        client_request = bytearray(self.default_header)
        client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        client_request.extend(payload_bytes)
        header = {"Authorization": f"Bearer; {self.token}"}
        async with websockets.connect(self.api_url, extra_headers=header, ping_interval=None) as ws:
            await ws.send(client_request)
            while True:
                msg = await ws.recv()
                done = self.__response(msg, filepath, async_play)
                if done:
                    break
            LOGGER.info("closing the connection...")
        byte_data = None
        with open(filepath, 'rb') as f:
            byte_data = f.read()
        return byte_data

    def __response(self, msg, filepath, async_play):
        header_size = msg[0] & 0x0f
        message_type = msg[1] >> 4
        message_type_specific_flags = msg[1] & 0x0f
        message_compression = msg[2] & 0x0f
        payload = msg[header_size*4:]
        if message_type == 0xb:
            if message_type_specific_flags == 0:
                return False
            else:
                sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                payload = payload[8:]
            if async_play:
                self.handler.play_audio(payload)
            if self.start_time is not None:
                end_time = time.time()
                LOGGER.info(f"TTS Volcengine 首帧耗时: {end_time - self.start_time:.4f} 秒")
                self.start_time = None
            with open(filepath, 'ab') as f:
                f.write(payload)
            if sequence_number < 0:
                return True
            else:
                return False
        elif message_type == 0xf:
            error_msg = payload[8:]
            if message_compression == 1:
                error_msg = gzip.decompress(error_msg)
            error_msg = str(error_msg, "utf-8")
            LOGGER.warning(f"Error message: {error_msg}")
            return True
        elif message_type == 0xc:
            payload = payload[4:]
            if message_compression == 1:
                payload = gzip.decompress(payload)
            LOGGER.warning(f"Frontend message: {payload}")
        else:
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
        
        filename = 'volcengine_'+self.md5(self.voice+text)+'.pcm'
        filepath = os.path.join(self.wav_dir,filename)
        
        audio_stream = None
        if os.path.exists(filepath) :
            with open(filepath, "rb") as f:
                audio_stream = f.read()  # 读取文件数据
            if async_play:
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
        self.to_audio(text,True)
