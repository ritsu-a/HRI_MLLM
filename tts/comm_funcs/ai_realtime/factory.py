#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-04-14
@author: wangzhongbin
'''

from .protocol import TurnDetectionType
from typing import Optional, Callable, List, Dict, Any
from llama_index.core.tools import BaseTool
from .zhipu import ZhiPu
from .openai import OpenAI

class Factory:
    '''
    ai工厂方法
    '''
    
    @staticmethod
    def get_agent(
        type,
        instructions: str = '',
        temperature: float = 0.8,
        turn_detection_type: TurnDetectionType = TurnDetectionType.CLIENT_VAD,
        tools: Optional[List[BaseTool]] = None,
        on_start_done: Optional[Callable[[], None]] = None,
        on_response_done: Optional[Callable[[], None]] = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_audio_delta: Optional[Callable[[bytes], None]] = None,
        on_audio_cleared: Optional[Callable[[], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_input_transcript: Optional[Callable[[str], None]] = None,
        on_output_transcript: Optional[Callable[[str], None]] = None,
        extra_event_handlers: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None,
    ):
        '''
        获取ai_realtime对象
        @param instructions: 系统提示词，用于指导模型行为
        @param temperature: 控制模型输出的随机性，值越高越“活跃”，越低越“稳重”
        @param turn_detection_type: 表示使用哪种说话结束检测方式（客户端/服务端）
        @param tools: 给模型提供可调用的函数（例如天气查询函数等）
        @param on_start_done: 回调函数：启动完成
        @param on_response_done: 回调函数：请求完成
        @param on_text_delta: 回调函数：接收到文本增量时调用（每次生成一段文本）
        @param on_audio_delta: 回调函数：接收到语音数据增量时调用（用于播放音频）
        @param on_audio_cleared: 回调函数：清除缓冲区中的音频
        @param on_interrupt: 回调函数：用户打断时调用（如说话中断）
        @param on_input_transcript: 回调函数：用户说话转录文本时调用
        @param on_output_transcript: 回调函数：模型回应的语音转文本
        @param extra_event_handlers: 扩展事件处理器，支持处理额外自定义事件类型
        @return ai_realtime对象
        '''
        if type == 'zhipu':
            return ZhiPu(instructions, temperature, turn_detection_type, tools, on_start_done, on_response_done, on_text_delta, on_audio_delta, on_audio_cleared, on_interrupt, on_input_transcript, on_output_transcript, extra_event_handlers)
        elif type == 'openai':
            return OpenAI(instructions, temperature, turn_detection_type, tools, on_start_done, on_response_done, on_text_delta, on_audio_delta, on_audio_cleared, on_interrupt, on_input_transcript, on_output_transcript, extra_event_handlers)
        else:
            raise ValueError("Invalid ai type")
