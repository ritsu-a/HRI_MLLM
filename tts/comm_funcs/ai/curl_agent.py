#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-02-28
@author: wangzhongbin, hejunfu
'''

import os
import json
import pycurl
from io import BytesIO
from .protocol import Protocol
from typing import List, Dict, Callable
from tts.common.logger import LOGGER,ExecutionTime,ExecutionError

class CurlAgent(Protocol):
    '''
    CurlAgent
    '''
    def __init__(self, model:str, host:str, api_key:str, timeout = 12):
        '''
        初始化
        @param model: 模型
        @param host: url
        @param api_key: key
        '''
        self.model = model
        self.host = host
        self.api_key = api_key
        self.timeout = timeout
    
    def chat(self, messages: List[Dict], stream: bool, callback: Callable):
        '''
        chat 推理
        @param messages: 消息 [{'role': 'user', 'content': content}],
        @param stream: 启用流式响应
        @param callback: 回调函数，用于在推理完成后处理结果
        '''
        uri = '/chat/completions'
        self.request(messages, stream, callback, {}, uri)

    def request(self, messages: List[Dict], stream: bool, callback: Callable[[bool,Dict], None], arguments: Dict = {}, url: str = ''):
        '''
        request 推理
        @param messages: 消息 [{'role': 'user', 'content': content}],
        @param stream: 启用流式响应
        @param callback: 回调函数，用于在推理完成后处理结果
        @param arguments: 消息参数，{'temperature':0.6}
        @param url: 接口地址
        '''
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(arguments)
        uri = f"{self.host}{url}"
        self.__request(uri, payload, stream, callback)

    @ExecutionError
    def __request(self, url: str, payload: Dict, stream: bool, callback: Callable[[bool,Dict], None]):
        '''
        curl 请求
        '''
        curl = None
        try:
            curl = pycurl.Curl()
            curl.setopt(curl.SSL_VERIFYPEER, False)  # 禁用 SSL 验证
            curl.setopt(curl.URL, url)  # 设置 URL
            curl.setopt(curl.HTTPHEADER, [
                "Content-Type: application/json",
                f"Authorization: Bearer {self.api_key}"  # 设置授权头
            ])
            curl.setopt(curl.POST, 1)  # 设置为 POST 请求
            curl.setopt(curl.POSTFIELDS, json.dumps(payload))  # 设置 POST 数据

            if stream:
                def stream_callback(chunk):
                    try:
                        decoded_data = chunk.decode('utf-8')
                        if decoded_data:
                            lines = decoded_data.splitlines()
                            for line in lines:
                                line = line.lstrip('data: ').strip() # 解码并去掉 'data: ' 前缀
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    callback(stream, data)  # 传递给 callback 进行处理
                                except json.JSONDecodeError:
                                    continue  # 如果某行解析失败，跳过
                    except Exception as e:
                        LOGGER.error(f"Error processing data: {e}")
                
                curl.setopt(curl.WRITEFUNCTION, stream_callback) # 使用 WRITEFUNCTION 选项设置回调函数
                curl.perform() # 执行请求
                self.__network_analysis(curl)
            else:
                buffer = BytesIO() # 使用 BytesIO 作为响应缓冲区
                curl.setopt(curl.WRITEDATA, buffer)  # 将响应数据写入 buffer
                curl.setopt(curl.TIMEOUT, self.timeout)           # 设置请求超时时间
                curl.perform() # 执行请求  
                self.__network_analysis(curl)

                if curl.getinfo(curl.RESPONSE_CODE) == 200: # 检查请求是否成功
                    response = buffer.getvalue().decode('utf-8')
                    callback(stream, json.loads(response))
                else:
                    LOGGER.error(f"Error: {curl.getinfo(curl.RESPONSE_CODE)}")
        except Exception as ex:
            LOGGER.error(f"curl chat 错误: {str(ex)}")
        finally:
            if curl != None:
                curl.close()
                
    def __network_analysis(self, curl: pycurl.Curl):
        '''
        网络分析
        '''
        if os.getenv("APP_ENV") == "debug" :
            # 获取耗时数据
            time_total = curl.getinfo(pycurl.TOTAL_TIME)  # 总时间
            time_namelookup = curl.getinfo(pycurl.NAMELOOKUP_TIME)  # DNS 解析时间
            time_connect = curl.getinfo(pycurl.CONNECT_TIME)  # TCP 连接时间
            time_appconnect = curl.getinfo(pycurl.APPCONNECT_TIME)  # SSL 握手时间
            time_pretransfer = curl.getinfo(pycurl.PRETRANSFER_TIME)  # 准备传输时间
            time_starttransfer = curl.getinfo(pycurl.STARTTRANSFER_TIME)  # 首字节时间
            time_redirect = curl.getinfo(pycurl.REDIRECT_TIME)  # 重定向时间

            # 打印耗时数据
            LOGGER.debug(f"Total Time: {time_total} seconds")
            LOGGER.debug(f"DNS Lookup Time: {time_namelookup} seconds")
            LOGGER.debug(f"TCP Connection Time: {time_connect} seconds")
            LOGGER.debug(f"SSL Handshake Time: {time_appconnect} seconds")
            LOGGER.debug(f"Pre-transfer Time: {time_pretransfer} seconds")
            LOGGER.debug(f"Start Transfer Time: {time_starttransfer} seconds")
            LOGGER.debug(f"Redirect Time: {time_redirect} seconds")