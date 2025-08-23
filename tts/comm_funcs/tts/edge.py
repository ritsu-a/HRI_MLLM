#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2024-05-17
@author: wangzhongbin
'''

import os
import time
import asyncio
import edge_tts
from io import BytesIO
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from tts.common.logger import LOGGER,ExecutionError
from .protocol import Protocol

class Edge(Protocol):
    '''
    edge 文字转音频类
    '''
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.voice = os.getenv("TTS_EDGE_VOICE")
    
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
        communicate = edge_tts.Communicate(text, self.voice)
        byte_data = b""
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
                if len(audio_bytes) >= 4096:
                    pcm_data = self.mp3_to_pcm(audio_bytes)
                    if async_play:
                        self.handler.play_audio(pcm_data)
                    if self.start_time is not None:
                        end_time = time.time()
                        LOGGER.info(f"TTS Edge 首帧耗时: {end_time - self.start_time:.4f} 秒")
                        self.start_time = None
                    with open(filepath, 'ab') as f:
                        f.write(pcm_data)
                    byte_data += pcm_data
                    audio_bytes = b""
                    
        if len(audio_bytes) > 0:
            pcm_data = self.mp3_to_pcm(audio_bytes)
            if async_play:
                self.handler.play_audio(pcm_data)
            with open(filepath, 'ab') as f:
                f.write(pcm_data)
            byte_data += pcm_data
        return byte_data
    
    def mp3_to_pcm(self, audio_bytes:bytes):
        audio = AudioSegment.from_mp3(BytesIO(audio_bytes))
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=300, silence_thresh=-50) # 检测非静音区段（避免前后杂音/静音）
        if nonsilent_ranges:
            start, end = nonsilent_ranges[0][0], nonsilent_ranges[-1][1]
            trimmed_audio = audio[start:end]
        else:
            trimmed_audio = audio  # 没识别出来，就全用
        wav_bytes = BytesIO()
        trimmed_audio.export(wav_bytes, format='wav')
        pcm_data = wav_bytes.getvalue()[44:]
        return pcm_data

    def to_audio(self, text:str, async_play:bool = False):
        '''
        文字转音频
        @param str: 文字
        @param async_play: 是否异步播放
        @return BytesIO: 返回音频
        '''
        if text is None or isinstance(text,str) == False or text == '':
            msg = 'text 是空的'
            LOGGER.warning(msg)
            return None
        
        filename = 'edge_'+self.md5(self.voice+text)+'.pcm'
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
        # print(122222222, audio_stream)
        return audio_stream
    
    def play(self, text:str):
        '''
        播放音频
        @param text: 文字
        '''
        self.to_audio(text, True)
       