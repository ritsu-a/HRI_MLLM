#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-04-05
@author: wangzhongbin
'''

import time
import os
import pyaudio
import webrtcvad
from pydub import AudioSegment
from .protocol import Protocol
from tts.common.logger import LOGGER
from typing import Optional, Callable

class VAD(Protocol):
    '''
    VAD 自动断句录音类
    '''

    def __init__(self, on_start_done: Optional[Callable[[], None]] = None,):
        super().__init__()
        self.format = pyaudio.paInt16 # 设置采样大小和格式
        self.vad = webrtcvad.Vad(3)
        self.frame_duration = 30  # ms
        self.frame_size = int(self.rate * self.frame_duration / 1000)  # 每帧采样点数量
        self.frame_bytes = self.frame_size * 2  # 每个采样点2字节（16位）
        self.silence_threshold = float(os.getenv('VAD_SILENCE_THRESHOLD',-30))  # 静音阈值
        self.on_start_done = on_start_done
        self.is_running = False
        self.speech_started = False
        max_silence_sec = float(os.getenv('VAD_SILENCE_TIMEOUT',1))# 说话中静音超时时间
        self.max_silence_frames = max_silence_sec*1000/self.frame_duration
        no_valid_voice_timeout = float(os.getenv('VAD_NO_VALID_VOICE_TIMEOUT',5))#无效输入等待最大时间
        self.no_valid_voice_timeout = no_valid_voice_timeout 

   
    def start(self):
        '''
        开始录音
        @return bytes: 音频数据
        '''
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.frame_size)
        self.speech_started = False
        silence_count = 0
        max_silence_frames = self.max_silence_frames
        speech_buffer = b""
        
        sleep=self.frame_duration / 1000.0
        time_cout=5.0/sleep
        timeout=0
        silence_timeout=0
        self.is_running = True
        try:
            LOGGER.info("🎤 开始监听并发送音频数据...")
            wait_start = time.time()
            while self.is_running:
                frame = stream.read(self.frame_size, exception_on_overflow=False)
                is_speech = self.vad.is_speech(frame, self.rate)
                if is_speech:
                    if not self.speech_started:
                        LOGGER.info("🟢 说话检测中，开始录音...")
                        self.speech_started = True
                        speech_buffer = b""
                    speech_buffer += frame
                    silence_count = 0
                    timeout=0
                   
                else:
                    if self.speech_started:
                        if self.on_start_done:
                            self.on_start_done()
                        silence_count += 1
                        if silence_count > max_silence_frames:
                            audio_segment = AudioSegment(speech_buffer, frame_rate=self.rate, sample_width=p.get_sample_size(self.format), channels=self.channels)
                            if audio_segment.dBFS < self.silence_threshold:
                                LOGGER.info(f"当前音频为静音 {audio_segment.dBFS}")
                                self.speech_started = False
                                speech_buffer = b""
                                if time.time()-wait_start>self.no_valid_voice_timeout:
                                    LOGGER.info(f"⏰ 无效输入超时，结束录音")
                                    break
                                # silence_count = 0
                                # if silence_timeout > time_cout:
                                #     return None
                            else:
                                LOGGER.info(f"⚪ 说话结束 {audio_segment.dBFS}")
                                break
                    else:
                        if time.time()-wait_start>self.no_valid_voice_timeout:
                            LOGGER.info(f"⏰ 无输入超时，结束录音")
                            break
                    #     timeout+=1
                    #     if timeout>time_cout:
                    #         return None
                # silence_timeout+=1
                time.sleep(sleep)
                # time.sleep(self.frame_duration / 1000.0)
                
        except Exception as e:
            LOGGER.info(f"❗ 发送时发生错误: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.speech_started = False
            LOGGER.info(f"🛑  监听结束")
        return speech_buffer
    def stop(self):
        self.is_running = False
    
    def is_recording(self):
        return self.speech_started
    
