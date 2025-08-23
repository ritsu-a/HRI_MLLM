#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2024-03-08
@author: wangzhongbin
'''

import wave
import numpy as np
from abc import ABC, abstractmethod
from tts.common.logger import ExecutionError

class Protocol(ABC):
    '''
    录音类协议
    '''
    
    def __init__(self):
        self.format = np.int16 # 设置采样大小和格式
        self.channels = 1 # 设置通道数
        self.rate = 16000 # 设置采样率 

    @abstractmethod
    def start(self):
        '''
        开始录音并返回录音数据
        @return bytes: 音频数据
        '''
    
    @ExecutionError
    def write(self, filepath, audio):
        '''
        录音数据写文件
        @param filepath: 存放路径
        @param data: 音频数据
        @return bool: 返回写结果%s
        开始录音
        @return bytes: 音频数据
        '''
        if len(audio) > 0 :
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.format().itemsize)
                wf.setframerate(self.rate)
                wf.writeframes(audio)
    @abstractmethod
    def stop(self):
        '''
        结束录音
        @return 
        '''
    
    @abstractmethod
    def is_recording(self):
        '''
        是否正在录音
        @return bool: 是否正在录音
        '''
    
