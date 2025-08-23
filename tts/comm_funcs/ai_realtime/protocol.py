#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-04-14
@author: wangzhongbin
'''

from enum import Enum
from abc import ABC, abstractmethod

class TurnDetectionType(Enum):
    SERVER_VAD = "server_vad"
    CLIENT_VAD = "client_vad"

class Protocol(ABC):
    '''
    realtime_ai 类协议
    '''

    def __init__(self):
        pass
    
    @abstractmethod
    def start(self):
        '''
        启动
        '''
    
    @abstractmethod
    def close(self):
        '''
        关闭服务
        '''

    @abstractmethod
    def session_update(self):
        '''
        会话配置更新
        '''

    @abstractmethod
    def send_text(self, text: str):
        '''
        发送字符串
        @param text: 字符串
        '''

    @abstractmethod
    def send_audio(self, audio_bytes: bytes):
        '''
        发送音频
        @param audio_bytes: 二进制音频
        '''
        
    @abstractmethod
    def stream_audio(self, audio_chunk: bytes):
        '''
        发送音频流
        @param audio_bytes: 二进制音频
        '''

    @abstractmethod
    def clear_audio(self):
        '''
        清除缓冲区中的音频
        '''
    
    @abstractmethod
    def send_audio_image(self, audio_bytes: bytes, image_bytes: bytes):
        '''
        发送图片
        @param audio_bytes: 二进制音频
        @param image_bytes: 二进制图片
        '''
    
    @abstractmethod
    def send_image(self, image_bytes: bytes):
        '''
        发送图片
        @param image_bytes: 二进制图片
        '''

    @abstractmethod
    def create_response(self):
        '''
        创建模型回复
        '''

    @abstractmethod
    def cancel_response(self):
        '''
        取消模型调用
        '''