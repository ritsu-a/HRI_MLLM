#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2024-05-17
@author: wangzhongbin
'''

import os
import hashlib
from abc import ABC, abstractmethod
from tts.common.logger import LOGGER
from audio.audio_handler import AudioHandler
import socket

class Protocol(ABC):
    '''
    文字转音频类协议
    '''

    def __init__(self):
        # self.wav_dir = os.getenv("WAV_DIR")
        self.wav_dir = os.path.dirname(os.path.abspath(__file__)) + "/record"  
        if not os.path.exists(self.wav_dir):
            os.makedirs(self.wav_dir)
        self.filepath = None
        self.handler = AudioHandler()
        self.udp_client_socket = None
        self.__init_udp_client()
    def __del__(self):
        if self.udp_client_socket:
            self.udp_client_socket.close()
            self.udp_client_socket = None
    
    @abstractmethod
    def to_audio(self, text:str, async_play:bool = False):
        '''
        文字转音频
        @param str: 文字
        @param async_play: 是否异步播放
        @return bytes: 返回音频
        '''
    
    @abstractmethod
    def play(self, text:str):
        '''
        文字转音频
        @param str: 文字
        @return
        '''

    def md5(self, data):
        '''
        md5 加密
        @param data: 加密数据
        @return string: 返回加密字符
        '''
        md5_hash = hashlib.md5()
        md5_hash.update(data.encode('utf-8'))
        digest = md5_hash.hexdigest()
        return digest
    
    def __init_udp_client(self):
        '''
        初始化 udp 客户端
        @return
        '''
        try:
            # 创建 UDP 套接字
            self.udp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # 设置超时时间，防止长时间阻塞
            self.udp_client_socket.settimeout(5)
            self.udp_server_address = (os.getenv("PLAY_UDP_SERVER_IP",'127.0.0.1'), int(os.getenv("PLAY_UDP_SERVER_PORT",'5556')))
        except socket.error as socket_error:
            LOGGER.info(f"创建或操作套接字时出错: {socket_error}")
    def play_by_udp(self, data_bytes):
        try:
            # 发送数据
            # self.udp_client_socket.sendto(data_bytes, self.udp_server_address)
            # LOGGER.info(f"数据发送成功,len={len(data_bytes)}")
            # 每个数据块的最大长度
            chunk_size = 10240
            data_len = len(data_bytes)
            # 分割数据并发送
            for i in range(0, data_len, chunk_size):
                chunk = data_bytes[i:i + chunk_size]
                ret = self.udp_client_socket.sendto(chunk, self.udp_server_address)
                LOGGER.info(f"数据发送成功,len={len(data_bytes)}")

        except Exception as send_error:
            LOGGER.error(f"发送数据时出错: {send_error}")
