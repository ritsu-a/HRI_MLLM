#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-04-14
@author: wangzhongbin
'''

import os
import json
import base64
import websocket
import threading
from tts.common.logger import LOGGER
from .protocol import Protocol,TurnDetectionType
from typing import Optional, Callable, List, Dict, Any
from llama_index.core.tools import BaseTool, ToolSelection, adapt_to_async_tool, call_tool_with_selection

class OpenAI(Protocol):
    def __init__(
        self,
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
        super().__init__()
        self.api_key = os.getenv("AI_REALTIME_KEY")
        self.voice = os.getenv("AI_REALTIME_VOICE")
        self.instructions = instructions
        self.temperature = temperature
        self.turn_detection_type = turn_detection_type
        self.base_url = os.getenv("AI_REALTIME_URL")
        self.model = os.getenv("AI_REALTIME_MODEL")

        self.on_start_done = on_start_done
        self.on_response_done = on_response_done
        self.on_text_delta = on_text_delta
        self.on_audio_delta = on_audio_delta
        self.on_audio_cleared = on_audio_cleared
        self.on_interrupt = on_interrupt
        self.on_input_transcript = on_input_transcript
        self.on_output_transcript = on_output_transcript
        self.extra_event_handlers = extra_event_handlers or {}

        tools = tools or []
        self.tools: List[BaseTool] = [adapt_to_async_tool(t) for t in tools]

        self._is_start = False
        self._current_response_id = None
        self._current_item_id = None
        self._is_responding = False
        self._print_input_transcript = False
        self._output_transcript_buffer = ""

        self.ws_app = websocket.WebSocketApp(
            f"{self.base_url}?model={self.model}",
            header={"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"},
            on_message=self.__on_message,
            on_open=self.__on_open,
            on_error=self.__on_error,
            on_close=self.__on_close,
        )

    def start(self):
        '''
        启动
        '''
        self.thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.thread.start()

    def close(self):
        '''
        关闭服务
        '''
        self.ws_app.close()
        self._is_start = False

    def session_update(self):
        '''
        会话配置更新
        '''
        config = {
            "modalities": ["text", "audio"],
            "instructions": self.instructions,
            "voice": self.voice,
            "tools": [t.metadata.to_openai_tool()["function"] | {"type": "function"} for t in self.tools],
            "tool_choice": "auto",
            "temperature": self.temperature,
            "input_audio_transcription": {
                "model": "whisper-1"
            },
        }
        if self.turn_detection_type == TurnDetectionType.SERVER_VAD:
            config["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 500,
                "silence_duration_ms": 200
            }
            config["input_audio_format"] = "pcm16"
            config["output_audio_format"] = "pcm16"
        else:
            config["input_audio_format"] = "pcm16"
            config["output_audio_format"] = "pcm16"
        self.__send_event({
            "type": "session.update",
            "session": config,
        })

    def send_text(self, text: str):
        '''
        发送字符串
        @param text: 字符串
        '''
        if self._is_start == False:
            raise('请启动后再调用')
        
        self.__send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        })
        if self.turn_detection_type ==TurnDetectionType.CLIENT_VAD:
            self.create_response()

    def send_audio(self, audio_bytes: bytes):
        '''
        发送音频
        @param audio_bytes: 二进制音频
        '''
        if self._is_start == False:
            raise('请启动后再调用')
        
        audio_b64 = base64.b64encode(audio_bytes).decode()
        self.__send_event({"type": "input_audio_buffer.append", "audio": audio_b64})
        self.__send_event({"type": "input_audio_buffer.commit"})
        if self.turn_detection_type == TurnDetectionType.CLIENT_VAD:
            self.create_response()
        
    def stream_audio(self, audio_chunk: bytes):
        '''
        发送音频流
        @param audio_bytes: 二进制音频
        '''
        if self._is_start == False:
            raise('请启动后再调用')
        audio_b64 = base64.b64encode(audio_chunk).decode()
        self.__send_event({"type": "input_audio_buffer.append", "audio": audio_b64})

    def clear_audio(self):
        '''
        清除缓冲区中的音频
        '''
        if self._is_start == False:
            raise('请启动后再调用')
        self.__send_event({"type": "input_audio_buffer.clear"})

    def send_audio_image(self, audio_bytes: bytes, image_bytes: bytes):
        '''
        发送图片
        @param audio_bytes: 二进制音频
        @param image_bytes: 二进制图片
        '''
        if self._is_start == False:
            raise('请启动后再调用')
        self.send_audio(audio_bytes)

    def send_image(self, image_bytes: bytes):
        '''
        发送图片
        @param image_bytes: 二进制图片
        '''
        if self._is_start == False:
            raise('请启动后再调用')
        pass

    def create_response(self):
        '''
        创建模型回复
        '''
        if self._is_start == False:
            raise('请启动后再调用')
        
        self.__send_event({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"]
            }
        })

    def cancel_response(self):
        '''
        取消模型调用 response.cancel
        '''
        if self._is_start == False:
            raise('请启动后再调用')
        
        self.__send_event({"type": "response.cancel"})

    def __handle_interruption(self):
        if not self._is_responding:
            return
        LOGGER.info("\n[Handling interruption]")
        if self._current_response_id:
            self.cancel_response()
        if self._current_item_id:
            self.__truncate_response()
        self._is_responding = False
        self._current_response_id = None
        self._current_item_id = None

    def __truncate_response(self):
        if self._current_item_id:
            self.__send_event({"type": "conversation.item.truncate", "item_id": self._current_item_id})

    def __call_tool(self, call_id: str, tool_name: str, tool_args: Dict[str, Any]):
        tool_selection = ToolSelection(tool_id="tool_id", tool_name=tool_name, tool_kwargs=tool_args)
        result = call_tool_with_selection(tool_selection, self.tools, verbose=True)
        self.__send_function_result(call_id, str(result))

    def __send_function_result(self, call_id: str, result: Any):
        self.__send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            }
        })
        self.create_response()

    def __send_event(self, data: Dict[str, Any]):
        try:
            msg = json.dumps(data)
            LOGGER.debug(msg)
            self.ws_app.send(msg)
        except Exception as e:
            LOGGER.warning(f"[Exception in __send_event] {str(e)}")

    def __on_open(self, ws):
        self._is_start = True
        self.session_update()
        if self.on_start_done:
            self.on_start_done()
        
    def __on_error(self, ws, error):
        LOGGER.warning(f"[WebSocket Error] {error}")

    def __on_close(self, ws, code, msg):
        self._is_start = False
        LOGGER.info(f"[WebSocket Closed] code={code}, reason={msg}")

    def __on_message(self, ws, message):
        try:
            event = json.loads(message)
            event_type = event.get("type")
            # LOGGER.debug(f"[type] {event_type}")
            LOGGER.debug(f"[message] {message}")
            if event_type == "error":
                LOGGER.info(f"[Error] {event['error']}")
            elif event_type == "response.created":
                self._current_response_id = event.get("response", {}).get("id")
                self._is_responding = True
            elif event_type == "response.output_item.added":
                self._current_item_id = event.get("item", {}).get("id")
            elif event_type == "input_audio_buffer.cleared":
                self.on_audio_cleared()
            elif event_type == "response.done":
                self._is_responding = False
                self._current_response_id = None
                self._current_item_id = None
                if self.on_response_done:
                    self.on_response_done()
            elif event_type == "input_audio_buffer.speech_started":
                LOGGER.info("[Speech Started]")
                if self._is_responding:
                    self.__handle_interruption()
                if self.on_interrupt:
                    self.on_interrupt()
            elif event_type == "input_audio_buffer.speech_stopped":
                LOGGER.info("\n[Speech ended]")
            elif event_type == "response.text.delta" and self.on_text_delta:
                self.on_text_delta(event["delta"])
            elif event_type == "response.audio.delta" and self.on_audio_delta:
                self.on_audio_delta(base64.b64decode(event["delta"]))
            elif event_type == "response.function_call_arguments.done":
                self.__call_tool(event["call_id"],event["name"],json.loads(event["arguments"]))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                if self.on_input_transcript:
                    self.on_input_transcript(event.get("transcript", ""))
                    self._print_input_transcript = True
            elif event_type == "response.audio_transcript.delta":
                delta = event.get("delta", "")
                if self.on_output_transcript:
                    if not self._print_input_transcript:
                        self._output_transcript_buffer += delta
                    else:
                        if self._output_transcript_buffer:
                            self.on_output_transcript(self._output_transcript_buffer)
                            self._output_transcript_buffer = ""
                        self.on_output_transcript(delta)
            elif event_type == "response.audio_transcript.done":
                self._print_input_transcript = False
            elif event_type in self.extra_event_handlers:
                self.extra_event_handlers[event_type](event)
        except Exception as e:
            LOGGER.warning(f"[Exception in message handler] {str(e)}")
