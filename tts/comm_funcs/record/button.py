#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-04-15
@author: wangzhongbin
'''

import os
import time
import threading
import pyaudio
import serial
import serial.tools.list_ports
from .protocol import Protocol
from tts.common.logger import LOGGER
from typing import Optional, Callable

class Button(Protocol):
    '''
    按键录音（使用 pyaudio）
    '''

    def __init__(self, on_start_done: Optional[Callable[[], None]] = None,):
        super().__init__()
        self.listening = False
        self.ser = None
        self.format = pyaudio.paInt16
        self.frame_duration = 30  # ms
        self.frame_size = int(self.rate * self.frame_duration / 1000)  # 每帧采样点数量
        self.on_start_done = on_start_done
        self.is_running = False

        # 查找串口设备
        device_tag = os.getenv('RECORD_DEVICE_TAG')  # 唯一确定设备名称
        ports_list = list(serial.tools.list_ports.comports())
        for device in ports_list:
            if list(device)[1] == device_tag:
                try:
                    self.ser = serial.Serial(list(device)[0], 9600)
                    if self.ser.is_open:
                        LOGGER.info("串口已打开")
                except Exception as e:
                    LOGGER.warning(f"Serial port error: {e}")

    def __close(self):
        while self.is_running:
            if self.ser.in_waiting:
                data = self.ser.read(self.ser.in_waiting)
                if data == b'AA':
                    LOGGER.info("收到停止指令 AA")
                    # self.ser.close()
                    self.ser.flush()
                    self.listening = False
                    return
            time.sleep(0.1)

    def start(self):
        '''
        开始录音
        @return bytes: 音频数据
        '''
        self.listening = False
        self.is_running = True
        LOGGER.info("等待按键启动录音")
        while self.is_running:
            if self.ser.in_waiting:
                data = self.ser.read(self.ser.in_waiting)
                if data == b'55':
                    LOGGER.info("开始录音")
                    self.listening = True
                    break
            time.sleep(0.1)

        if self.on_start_done:
            self.on_start_done()
        
        # 启动监听线程
        button_thread = threading.Thread(target=self.__close)
        button_thread.start()
        LOGGER.info("再次按下按键发送 AA 以停止录音")

        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.frame_size)
        speech_buffer = b""
        sleep = self.frame_duration / 1000.0
        try:
            while self.listening:
                audio_chunk = stream.read(self.frame_size, exception_on_overflow=False)
                speech_buffer += audio_chunk
                time.sleep(sleep)
        except Exception as e:
            LOGGER.warning(f"录音错误: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        LOGGER.info("录音结束")
        return speech_buffer
    def stop(self):
        self.is_running = False
        self.listening = False
    
    def is_recording(self):
        return self.listening




