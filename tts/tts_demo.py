import os
import sys
from typing import Optional, Dict
from dotenv import load_dotenv
from comm_funcs.tts import Factory as TTS_Factory #TTS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pydub import AudioSegment

# 测试代码
if __name__ == "__main__":
    load_dotenv()
    say = "你好，我在银河通用机器人公司！"
    tts = TTS_Factory.get_tts(os.getenv("TTS_TYPE"))  #TTS
    # tts.play(say)
    audio_bytes = tts.to_audio(say, async_play = False)

    if audio_bytes:
       # 用 pydub 直接封装成 WAV；参数从 handler 读取，避免硬编码
        pcm = AudioSegment(
            audio_bytes,
            sample_width=tts.handler.audio.get_sample_size(tts.handler.format),
            frame_rate=tts.handler.rate,
            channels=tts.handler.channels
        )
        pcm.export("tts_0823.wav", format="wav")
        print("✅ 已保存: tts.wav")
    else:
        print("⚠️ 生成音频失败：audio_bytes 为空")

    # tts.handler.wait_for_playback_complete()