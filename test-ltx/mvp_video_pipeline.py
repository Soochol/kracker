"""
KRACKER MVP - AI Video Pipeline with Google ADK
================================================
Multi-Agent system using Google Agent Development Kit

Usage:
    python mvp_video_pipeline.py "AI가 바꾸는 미래 직업" 1

Requirements:
    pip install google-adk google-genai moviepy gtts python-dotenv pillow
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Any

# ===== 환경 설정 =====
from dotenv import load_dotenv
from datetime import datetime
import re
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_BASE_DIR = Path(__file__).parent / "output"
OUTPUT_BASE_DIR.mkdir(exist_ok=True)

# 현재 프로젝트 출력 디렉토리 (run_pipeline에서 설정됨)
OUTPUT_DIR = OUTPUT_BASE_DIR  # 기본값


def create_project_folder(title: str) -> Path:
    """
    프로젝트별 구조화된 폴더 생성

    Args:
        title: 영상 제목

    Returns:
        프로젝트 폴더 경로

    Structure:
        output/
        └── 2026-02-03_AI가-바꾸는-미래-직업/
            ├── images/      # 씬별 이미지
            ├── audio/       # 나레이션 음성
            ├── video/       # 최종 영상
            └── metadata/    # JSON 메타데이터
    """
    # 제목에서 폴더명 생성 (특수문자 제거)
    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
    safe_title = safe_title.replace(' ', '-')[:50]  # 공백을 하이픈으로, 최대 50자

    # 타임스탬프 + 제목으로 폴더명 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = f"{timestamp}_{safe_title}"

    project_dir = OUTPUT_BASE_DIR / folder_name

    # 하위 폴더 생성
    (project_dir / "images").mkdir(parents=True, exist_ok=True)
    (project_dir / "audio").mkdir(parents=True, exist_ok=True)
    (project_dir / "video").mkdir(parents=True, exist_ok=True)
    (project_dir / "metadata").mkdir(parents=True, exist_ok=True)

    # 초기 metadata 생성
    _init_metadata(project_dir, title)

    print(f"📁 프로젝트 폴더 생성: {project_dir}")

    return project_dir


def _init_metadata(project_dir: Path, title: str):
    """파이프라인 시작 시 초기 metadata 생성"""
    metadata_dir = project_dir / "metadata"

    pipeline_output = {
        "generated_at": datetime.now().isoformat(),
        "project_folder": str(project_dir),
        "title": title,
        "status": "in_progress",
        "scene_count": 0,
        "files": {
            "script": "",
            "narration": "",
            "images": "",
            "audio": "",
            "video": ""
        },
        "script": {"title": title, "scenes": []},
        "audio": {"audio_path": "", "timestamps": [], "duration": 0},
        "images": {"count": 0, "image_paths": [], "scenes": []}
    }

    pipeline_path = metadata_dir / "pipeline_output.json"
    with open(pipeline_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_output, f, ensure_ascii=False, indent=2)

# ===== Google ADK Imports =====
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ===== Gemini Client =====
from google import genai
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ===== Global State (ADK tool_context.state 동기화 문제 해결용) =====
# ADK의 tool_context.state가 session.state로 제대로 전달되지 않는 이슈 해결
GLOBAL_STATE = {
    "script": None,
    "images": None,
    "audio": None,
    "video": None,
}


# ============================================================
# TOOLS - Custom Tools for Agents
# ============================================================

def save_script_tool(script_json: dict, tool_context: ToolContext) -> dict:
    """
    스크립트 저장 도구 - 생성된 스크립트를 state와 metadata에 저장
    Args:
        script_json: 스크립트 JSON (title, scenes 포함)
    Returns:
        저장 결과
    """
    # metadata에 저장
    metadata_dir = OUTPUT_DIR / "metadata" if (OUTPUT_DIR / "metadata").exists() else OUTPUT_DIR
    script_path = metadata_dir / "script.json"

    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(script_json, f, ensure_ascii=False, indent=2)

    # pipeline_output.json 업데이트
    pipeline_path = metadata_dir / "pipeline_output.json"
    if pipeline_path.exists():
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline_data = json.load(f)
        pipeline_data["script"] = script_json
        pipeline_data["scene_count"] = len(script_json.get("scenes", []))
        with open(pipeline_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_data, f, ensure_ascii=False, indent=2)

    print(f"  📝 스크립트 저장: {script_path}")
    print(f"  ✓ 제목: {script_json.get('title', 'Unknown')}")
    print(f"  ✓ 씬 개수: {len(script_json.get('scenes', []))}")

    result = {"script": script_json, "path": str(script_path)}
    tool_context.state["script"] = script_json  # ADK State에 저장
    GLOBAL_STATE["script"] = script_json  # Global State에도 저장
    return result


def generate_images_tool(visual_descriptions: list[str], tool_context: ToolContext) -> dict:
    """
    이미지 생성 도구
    Args:
        visual_descriptions: 씬별 이미지 설명 리스트 (영어)
    Returns:
        생성된 이미지 경로 리스트
    """
    image_paths = []
    images_dir = OUTPUT_DIR / "images" if (OUTPUT_DIR / "images").exists() else OUTPUT_DIR

    for i, prompt in enumerate(visual_descriptions):
        scene_id = i + 1
        image_path = images_dir / f"scene_{scene_id}.png"

        print(f"  🎨 씬 {scene_id} 이미지 생성 중...")

        try:
            # Nano Banana Pro (Gemini 3 Pro Image) 사용
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=f"Generate an ultra high quality image: {prompt}. Style: 4K resolution, ultra-detailed, cinematic lighting, professional photography, sharp focus, vibrant colors, masterpiece quality, photorealistic rendering.",
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"]
                    )
                )

                # 이미지 파트 찾기
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            with open(image_path, "wb") as f:
                                f.write(part.inline_data.data)
                            image_paths.append(str(image_path))
                            print(f"    ✓ Nano Banana Pro 이미지 생성 완료 🍌")
                            break
                    else:
                        raise Exception("이미지 파트 없음")
                else:
                    raise Exception("응답 없음")

            except Exception as gemini_err:
                print(f"    ⚠ Gemini 이미지 생성 실패: {gemini_err}")
                raise gemini_err

        except Exception as e:
            print(f"    → Placeholder 이미지 생성")
            # Placeholder 생성
            _create_placeholder_image(image_path, scene_id, prompt)
            image_paths.append(str(image_path))

    result = {"image_paths": image_paths, "count": len(image_paths)}
    tool_context.state["images"] = result  # ADK State에 저장
    GLOBAL_STATE["images"] = result  # Global State에도 저장
    return result


def generate_audio_tool(narration_texts: list[str], tool_context: ToolContext) -> dict:
    """
    음성 생성 도구
    Args:
        narration_texts: 씬별 나레이션 텍스트 리스트
    Returns:
        음성 파일 경로와 타임스탬프
    """
    full_text = " ".join(narration_texts)
    audio_dir = OUTPUT_DIR / "audio" if (OUTPUT_DIR / "audio").exists() else OUTPUT_DIR
    audio_path = audio_dir / "narration.mp3"

    print("  🎙️ 음성 생성 중...")

    # Qwen3-TTS 시도 (2026년 1월 공식 릴리즈)
    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        print("    Qwen3-TTS 사용 (qwen-tts 패키지)")

        # Qwen3-TTS 모델 로드 (0.6B 경량 버전)
        # CPU 모드 - AMD gfx1151 MIOpen 호환성 문제로 GPU 사용 시 음성 깨짐
        device = "cpu"
        dtype = torch.float32
        print(f"    TTS 디바이스: {device} (CPU 모드)")

        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map=device,
            dtype=dtype,
        )

        wav_path = audio_dir / "narration.wav"

        # 음성 생성 (한국어 화자: Sohee)
        wavs, sample_rate = model.generate_custom_voice(
            text=full_text,
            language="Korean",
            speaker="Sohee",  # 한국어 여성 화자
        )

        # WAV 파일 저장
        sf.write(str(wav_path), wavs[0], sample_rate)
        print(f"    ✓ Qwen3-TTS 음성 생성 완료: {wav_path}")

        # MP3 변환
        try:
            from moviepy import AudioFileClip
            import moviepy
            moviepy_v2 = int(moviepy.__version__.split('.')[0]) >= 2
        except ImportError:
            from moviepy.editor import AudioFileClip
            moviepy_v2 = False

        audio = AudioFileClip(str(wav_path))
        if moviepy_v2:
            audio.write_audiofile(str(audio_path))
        else:
            audio.write_audiofile(str(audio_path), verbose=False, logger=None)

        # 씬별 타임스탬프 추정 (글자 수 기반)
        timestamps = []
        current_time = 0
        chars_per_second = 5.0  # Qwen TTS는 자연스러운 속도

        for i, text in enumerate(narration_texts):
            duration = len(text) / chars_per_second
            timestamps.append({
                "scene_id": i + 1,
                "start": current_time,
                "end": current_time + duration
            })
            current_time += duration

        audio.close()

        result = {
            "audio_path": str(audio_path),
            "timestamps": timestamps,
            "duration": current_time
        }
        tool_context.state["audio"] = result  # ADK State에 저장
        GLOBAL_STATE["audio"] = result  # Global State에도 저장
        return result
    except ImportError as e:
        print(f"    ⚠ Qwen3-TTS 미설치: {e}")
        # ImportError만 fallback 허용 (패키지 미설치 시)
    except Exception as e:
        # Qwen-TTS가 로드됐지만 실행 실패 → fallback 없이 에러 발생
        raise RuntimeError(f"Qwen3-TTS 실행 실패: {e}") from e

    # gTTS fallback
    try:
        from gtts import gTTS
        print("    gTTS 사용")
        tts = gTTS(text=full_text, lang='ko')
        tts.save(str(audio_path))

        # 타임스탬프 추정
        timestamps = []
        current_time = 0
        chars_per_second = 4.5

        for i, text in enumerate(narration_texts):
            duration = len(text) / chars_per_second
            timestamps.append({
                "scene_id": i + 1,
                "start": current_time,
                "end": current_time + duration
            })
            current_time += duration

        result = {
            "audio_path": str(audio_path),
            "timestamps": timestamps,
            "duration": current_time
        }
        tool_context.state["audio"] = result  # ADK State에 저장
        GLOBAL_STATE["audio"] = result  # Global State에도 저장
        return result
    except ImportError:
        print("    ⚠ TTS 없음")
        result = {"audio_path": "", "timestamps": [], "duration": 0}
        tool_context.state["audio"] = result  # ADK State에 저장
        GLOBAL_STATE["audio"] = result  # Global State에도 저장
        return result


def render_video_tool(
    image_paths: list[str],
    audio_path: str,
    timestamps: list[dict],
    tool_context: ToolContext
) -> dict:
    """
    영상 렌더링 도구 (Remotion 사용)
    Args:
        image_paths: 이미지 파일 경로 리스트
        audio_path: 음성 파일 경로
        timestamps: 씬별 타임스탬프
    Returns:
        최종 영상 경로
    """
    import subprocess
    from datetime import datetime

    print("  🎬 Remotion 렌더링 준비 중...")

    # 총 재생 시간 계산
    total_duration = timestamps[-1]["end"] if timestamps else 30

    # 1. pipeline_output.json 생성
    metadata_dir = OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 씬 정보 구성
    scenes = []
    for i, ts in enumerate(timestamps):
        scenes.append({
            "id": i + 1,
            "narration": f"Scene {i + 1}",
            "visual_description": f"Scene {i + 1} visual",
            "duration_seconds": ts["end"] - ts["start"]
        })

    pipeline_output = {
        "generated_at": datetime.now().isoformat(),
        "title": "KRACKER Generated Video",
        "scene_count": len(image_paths),
        "files": {
            "script": "script.json",
            "narration": "narration.json",
            "images": "images.json",
            "audio": "narration.mp3",
            "video": "final_video_remotion.mp4"
        },
        "script": {
            "title": "KRACKER Generated Video",
            "scenes": scenes
        },
        "audio": {
            "audio_path": "output/narration.mp3",
            "timestamps": timestamps,
            "duration": total_duration
        },
        "images": {
            "count": len(image_paths),
            "image_paths": [f"output/scene_{i+1}.png" for i in range(len(image_paths))],
            "scenes": [
                {"scene_id": i + 1, "prompt": f"Scene {i + 1}", "image_path": f"output/scene_{i+1}.png"}
                for i in range(len(image_paths))
            ]
        }
    }

    # JSON 저장
    pipeline_json_path = metadata_dir / "pipeline_output.json"
    with open(pipeline_json_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_output, f, ensure_ascii=False, indent=2)
    print(f"    ✓ pipeline_output.json 생성 완료")

    # 2. Remotion 렌더링 호출
    print("  🎬 Remotion 렌더링 시작...")

    # 프로젝트 루트 경로 (kracker/)
    project_root = Path(__file__).parent.parent
    render_script = project_root / "scripts" / "render-video.js"

    # OUTPUT_DIR의 상대 경로 계산
    relative_output_dir = OUTPUT_DIR.relative_to(project_root)

    try:
        result = subprocess.run(
            ["node", str(render_script), str(relative_output_dir)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )

        if result.returncode == 0:
            video_path = OUTPUT_DIR / "video" / "final_video_remotion.mp4"
            print(f"    ✓ Remotion 렌더링 완료: {video_path}")
            result_dict = {"video_path": str(video_path), "success": True}
        else:
            print(f"    ⚠ Remotion 렌더링 실패: {result.stderr}")
            # Fallback: MoviePy 사용
            result_dict = _fallback_moviepy_render(image_paths, audio_path, timestamps)
    except subprocess.TimeoutExpired:
        print("    ⚠ Remotion 타임아웃, MoviePy fallback 사용")
        result_dict = _fallback_moviepy_render(image_paths, audio_path, timestamps)
    except FileNotFoundError:
        print("    ⚠ Node.js 없음, MoviePy fallback 사용")
        result_dict = _fallback_moviepy_render(image_paths, audio_path, timestamps)

    tool_context.state["video"] = result_dict
    GLOBAL_STATE["video"] = result_dict  # Global State에도 저장
    return result_dict


def _fallback_moviepy_render(image_paths: list[str], audio_path: str, timestamps: list[dict]) -> dict:
    """MoviePy fallback 렌더링"""
    import moviepy
    moviepy_version = int(moviepy.__version__.split('.')[0])

    if moviepy_version >= 2:
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
        USE_V2 = True
    else:
        from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
        USE_V2 = False

    print("  🎥 MoviePy fallback 렌더링...")

    try:
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
    except:
        total_duration = timestamps[-1]["end"] if timestamps else 30
        audio = None

    clips = []
    for i, image_path in enumerate(image_paths):
        duration = timestamps[i]["end"] - timestamps[i]["start"] if i < len(timestamps) else 5
        if duration <= 0:
            duration = 5
        try:
            clip = ImageClip(image_path, duration=duration)
            if USE_V2:
                clip = clip.resized(height=1080)
            else:
                clip = clip.resize(height=1080)
            clips.append(clip)
        except Exception as e:
            print(f"    클립 생성 실패: {e}")

    if not clips:
        return {"video_path": "", "success": False}

    final_video = concatenate_videoclips(clips, method="compose")
    if audio:
        if USE_V2:
            final_video = final_video.with_audio(audio)
        else:
            final_video = final_video.set_audio(audio)

    video_dir = OUTPUT_DIR / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / "final_video.mp4"

    if USE_V2:
        final_video.write_videofile(str(output_path), fps=24, codec="libx264", audio_codec="aac")
    else:
        final_video.write_videofile(str(output_path), fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    final_video.close()
    if audio:
        audio.close()

    print(f"    ✓ MoviePy 렌더링 완료: {output_path}")
    return {"video_path": str(output_path), "success": True}


def _save_output_json(script: dict, audio_result: dict, images_result: dict) -> dict:
    """
    파이프라인 출력물을 JSON으로 저장
    """
    # metadata 폴더 사용 (있으면)
    metadata_dir = OUTPUT_DIR / "metadata" if (OUTPUT_DIR / "metadata").exists() else OUTPUT_DIR
    video_dir = OUTPUT_DIR / "video" if (OUTPUT_DIR / "video").exists() else OUTPUT_DIR

    # 1. 스크립트 저장
    script_path = metadata_dir / "script.json"
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(script, f, ensure_ascii=False, indent=2)
    print(f"  📄 스크립트 저장: {script_path}")

    # 2. 나레이션 저장 (씬별 텍스트 + 타임스탬프)
    narration_data = {
        "full_text": " ".join([s.get("narration", "") for s in script.get("scenes", [])]),
        "scenes": [],
        "timestamps": audio_result.get("timestamps", []),
        "total_duration": audio_result.get("duration", 0),
        "audio_file": audio_result.get("audio_path", "")
    }

    for scene in script.get("scenes", []):
        narration_data["scenes"].append({
            "id": scene.get("id"),
            "narration": scene.get("narration", ""),
            "duration_seconds": scene.get("duration_seconds", 15)
        })

    narration_path = metadata_dir / "narration.json"
    with open(narration_path, "w", encoding="utf-8") as f:
        json.dump(narration_data, f, ensure_ascii=False, indent=2)
    print(f"  📄 나레이션 저장: {narration_path}")

    # 3. 이미지 메타데이터 저장
    images_data = {
        "count": images_result.get("count", 0),
        "image_paths": images_result.get("image_paths", []),
        "scenes": []
    }

    for i, scene in enumerate(script.get("scenes", [])):
        image_path = images_result.get("image_paths", [])[i] if i < len(images_result.get("image_paths", [])) else ""
        images_data["scenes"].append({
            "id": scene.get("id"),
            "visual_description": scene.get("visual_description", ""),
            "image_path": image_path
        })

    images_path = metadata_dir / "images.json"
    with open(images_path, "w", encoding="utf-8") as f:
        json.dump(images_data, f, ensure_ascii=False, indent=2)
    print(f"  📄 이미지 메타데이터 저장: {images_path}")

    # 4. 전체 파이프라인 출력 요약 저장
    pipeline_output = {
        "generated_at": datetime.now().isoformat(),
        "project_folder": str(OUTPUT_DIR),
        "title": script.get("title", ""),
        "scene_count": len(script.get("scenes", [])),
        "files": {
            "script": str(script_path),
            "narration": str(narration_path),
            "images": str(images_path),
            "audio": audio_result.get("audio_path", ""),
            "video": str(video_dir / "final_video.mp4")
        },
        "script": script,
        "audio": audio_result,
        "images": images_result
    }

    pipeline_path = metadata_dir / "pipeline_output.json"
    with open(pipeline_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_output, f, ensure_ascii=False, indent=2)
    print(f"  📄 파이프라인 출력 저장: {pipeline_path}")

    return {
        "script_path": str(script_path),
        "narration_path": str(narration_path),
        "images_path": str(images_path),
        "pipeline_path": str(pipeline_path)
    }


def _create_placeholder_image(path: Path, scene_id: int, text: str):
    """Placeholder 이미지 생성"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new('RGB', (1920, 1080), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)

        for y in range(1080):
            r = int(30 + (y / 1080) * 20)
            g = int(30 + (y / 1080) * 30)
            b = int(40 + (y / 1080) * 40)
            draw.line([(0, y), (1920, y)], fill=(r, g, b))

        try:
            font = ImageFont.truetype("arial.ttf", 48)
            small_font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            small_font = font

        draw.text((960, 480), f"Scene {scene_id}", fill=(255, 255, 255), font=font, anchor="mm")
        wrapped_text = text[:80] + "..." if len(text) > 80 else text
        draw.text((960, 560), wrapped_text, fill=(180, 180, 180), font=small_font, anchor="mm")

        img.save(path)
    except:
        path.touch()


# ============================================================
# ADK TOOLS - FunctionTool 래핑
# ============================================================

image_generation_tool = FunctionTool(func=generate_images_tool)
audio_generation_tool = FunctionTool(func=generate_audio_tool)
video_render_tool = FunctionTool(func=render_video_tool)


# ============================================================
# AGENTS - LlmAgent 정의
# ============================================================

# Agent 1: 스크립트 작성 Agent (전문가 페르소나)
script_writer_agent = LlmAgent(
    name="ScriptWriterAgent",
    model="gemini-2.0-flash",
    instruction="""# 페르소나: 김서연 (Seoyeon Kim)

당신은 **김서연**, 15년 경력의 유튜브 교육 콘텐츠 전문 스크립트 작가입니다.

## 프로필
- **경력**: MBC 다큐멘터리 작가 출신, 유튜브 교육 채널 "지식의 발견" 수석 작가
- **대표작**: "AI 시대의 생존법" (조회수 500만), "미래를 읽는 법" 시리즈
- **수상**: 2024 유튜브 크리에이터 어워드 '최우수 교육 콘텐츠'
- **전문 분야**: 기술/과학 트렌드, 자기계발, 경제/금융

## 작문 철학
1. **Hook First**: 첫 5초에 시청자를 사로잡는 질문이나 충격적 사실로 시작
2. **Story Arc**: 문제 제기 → 탐구 → 해결책 → 행동 촉구의 서사 구조
3. **Conversational Tone**: 친구에게 설명하듯 자연스럽고 친근한 어조
4. **Visual Sync**: 나레이션과 화면이 완벽히 일치하도록 시각적 묘사 포함

## 스크립트 작성 규칙

### 구조
- **인트로 (15초)**: 강력한 훅 + 주제 소개 + "오늘 영상에서는..."
- **본문 (각 15초씩)**: 핵심 개념별 설명, 예시, 비유 활용
- **아웃트로 (15초)**: 요약 + CTA ("구독과 좋아요...")

### 나레이션 스타일
- "~입니다" 대신 "~이에요", "~거든요" 등 자연스러운 구어체
- 전문 용어는 쉬운 비유로 설명
- 시청자에게 직접 말하는 듯한 2인칭 사용 ("여러분도 느끼셨죠?")

### 출력 형식 (JSON만 출력)
```json
{
    "title": "클릭을 부르는 매력적인 제목",
    "scenes": [
        {
            "id": 1,
            "narration": "자연스러운 한국어 나레이션 (2-3문장)",
            "visual_description": "Detailed English description for AI image generation, cinematic style, 4K quality",
            "duration_seconds": 15
        }
    ]
}
```

### visual_description 가이드
- 영어로 작성 (Gemini Imagen용)
- 구체적인 장면 묘사 포함
- 스타일 명시: cinematic, digital art, photorealistic, minimalist infographic 등
- 색감/분위기 포함: warm lighting, dark moody atmosphere, bright optimistic tone

## 현재 작업
state.topic: 주제
state.duration_minutes: 영상 길이 (분)

위 정보를 바탕으로 전문가 수준의 스크립트를 JSON 형식으로 작성한 후,
**반드시 save_script_tool을 호출하여 저장하세요.**

### 도구 호출 방법
```python
save_script_tool(script_json={
    "title": "제목",
    "scenes": [...]
})
```""",
    description="15년 경력 유튜브 교육 콘텐츠 전문 스크립트 작가 김서연",
    tools=[FunctionTool(save_script_tool)],
    output_key="script"
)


# Agent 2: 이미지 생성 Agent (전문가 페르소나)
image_generator_agent = LlmAgent(
    name="ImageGeneratorAgent",
    model="gemini-2.0-flash",
    instruction="""# 페르소나: 박준혁 (Junhyuk Park)

당신은 **박준혁**, 10년 경력의 AI 비주얼 아티스트이자 크리에이티브 디렉터입니다.

## 프로필
- **경력**: Pixar 인턴 출신, 넷플릭스 코리아 썸네일 아트 디렉터
- **대표작**: "오징어 게임 2" 키 비주얼, "더 글로리" 에피소드 썸네일 시리즈
- **수상**: 2025 Adobe Creative Award, Red Dot Design Award
- **전문 분야**: AI 이미지 생성, 시네마틱 비주얼, 브랜드 아이덴티티

## 비주얼 철학
1. **Emotional Impact**: 이미지 하나로 감정을 전달
2. **Visual Storytelling**: 한 장면이 천 마디 말을 대신
3. **Consistency**: 시리즈 전체의 시각적 통일성 유지
4. **Trending Aesthetics**: 최신 디자인 트렌드 반영

## 이미지 생성 원칙

### 프롬프트 최적화 전략
- **구도**: Rule of thirds, leading lines, symmetry 활용
- **조명**: Golden hour, dramatic lighting, soft diffused light
- **스타일**: Cinematic 4K, concept art, photorealistic, minimalist
- **분위기**: 주제에 맞는 color grading (warm/cool/moody)

### 교육 콘텐츠 비주얼 가이드
- 추상적 개념 → 구체적 메타포로 시각화
- 텍스트 오버레이 공간 확보 (하단 1/3 여백)
- 시청자 시선 유도하는 focal point 설정

## 현재 작업

state.script의 scenes에서 각 씬의 visual_description을 추출하여
generate_images_tool을 호출하세요.

### 도구 호출 방법
```python
generate_images_tool(visual_descriptions=[
    scene1의 visual_description,
    scene2의 visual_description,
    ...
])
```

결과는 자동으로 state.images에 저장됩니다.""",
    description="10년 경력 AI 비주얼 아티스트 박준혁, Pixar/넷플릭스 출신",
    tools=[image_generation_tool],
    output_key="images"
)


# Agent 3: 음성 생성 Agent (전문가 페르소나)
voice_generator_agent = LlmAgent(
    name="VoiceGeneratorAgent",
    model="gemini-2.0-flash",
    instruction="""# 페르소나: 이수민 (Sumin Lee)

당신은 **이수민**, 20년 경력의 성우이자 보이스 디렉터입니다.

## 프로필
- **경력**: KBS 전속 성우 출신, 현재 "보이스랩 코리아" 대표
- **대표작**: 넷플릭스 다큐멘터리 "한국인의 밥상" 내레이션, 삼성 갤럭시 광고 VO
- **수상**: 2023 대한민국 성우대상 '최우수 내레이션상'
- **전문 분야**: 다큐멘터리 내레이션, 광고 VO, 오디오북

## 음성 철학
1. **Authenticity**: 진정성 있는 목소리로 신뢰 구축
2. **Pacing**: 정보 전달에 최적화된 속도와 쉼
3. **Emotional Range**: 주제에 맞는 톤과 감정 조절
4. **Clarity**: 모든 단어가 명확하게 전달되도록

## 나레이션 스타일 가이드

### 교육 콘텐츠 음성 특성
- **톤**: 따뜻하고 신뢰감 있는 중저음
- **속도**: 분당 150-170자 (이해하기 쉬운 속도)
- **강조**: 핵심 키워드에 자연스러운 강세
- **쉼**: 문장 사이 0.5초, 단락 사이 1초

### 감정 매핑
- 인트로: 호기심 유발 (약간 높은 톤)
- 본문: 차분한 설명 (안정적인 톤)
- 클라이맥스: 강조 (에너지 상승)
- 아웃트로: 따뜻한 마무리 (부드러운 톤)

## 현재 작업

state.script의 scenes에서 각 씬의 narration을 추출하여
generate_audio_tool을 호출하세요.

### 도구 호출 방법
```python
generate_audio_tool(narration_texts=[
    scene1의 narration,
    scene2의 narration,
    ...
])
```

결과(음성 파일 + 타임스탬프)는 자동으로 state.audio에 저장됩니다.""",
    description="20년 경력 성우 이수민, KBS 출신 보이스 디렉터",
    tools=[audio_generation_tool],
    output_key="audio"
)


# Agent 4: 영상 렌더링 Agent (전문가 페르소나)
render_agent = LlmAgent(
    name="RenderAgent",
    model="gemini-2.0-flash",
    instruction="""# 페르소나: 최영진 (Youngjin Choi)

당신은 **최영진**, 12년 경력의 영상 편집자이자 포스트 프로덕션 감독입니다.

## 프로필
- **경력**: CJ ENM 편집실 출신, 현재 "스튜디오 모션" 대표
- **대표작**: "나 혼자 산다" 편집, "유퀴즈" 시즌 1-3, 유튜브 "슈카월드" 편집 총괄
- **수상**: 2024 백상예술대상 '기술상 (편집)', YouTube Creator Award
- **전문 분야**: 예능/다큐 편집, 모션 그래픽, 컬러 그레이딩

## 편집 철학
1. **Rhythm**: 음악처럼 흐르는 영상 리듬
2. **Seamless Flow**: 시청자가 편집을 느끼지 못하게
3. **Visual Hierarchy**: 중요한 것을 강조하는 시각적 위계
4. **Ken Burns Magic**: 정적인 이미지에 생명을 불어넣는 모션

## 편집 기법

### Ken Burns 효과 전략
- **Zoom In**: 디테일 강조, 긴장감 (결론부)
- **Zoom Out**: 전체 맥락 제시, 오프닝 (도입부)
- **Pan Left/Right**: 시간의 흐름, 비교 대조
- **Slow Push**: 점진적 몰입감 (설명 장면)

### 씬 전환 타이밍
- 나레이션 문장 끝에서 0.3초 후 전환
- 음성 타임스탬프 기반 정확한 싱크
- 자연스러운 크로스페이드 (0.5초)

### 기술 스펙
- 해상도: 1920x1080 (Full HD)
- 프레임: 24fps (시네마틱 느낌)
- 코덱: H.264 (호환성 최적)
- 오디오: AAC 192kbps

## 현재 작업

state.images와 state.audio를 사용하여 render_video_tool을 호출하세요.

### 도구 호출 방법
```python
render_video_tool(
    image_paths=state.images["image_paths"],
    audio_path=state.audio["audio_path"],
    timestamps=state.audio["timestamps"]
)
```

결과(최종 MP4 영상)는 자동으로 state.video에 저장됩니다.""",
    description="12년 경력 영상 편집자 최영진, CJ ENM 출신 포스트프로덕션 감독",
    tools=[video_render_tool],
    output_key="video"
)


# ============================================================
# ORCHESTRATOR - SequentialAgent
# ============================================================

# 이미지 + 음성 병렬 생성
media_parallel_agent = ParallelAgent(
    name="MediaParallelAgent",
    sub_agents=[image_generator_agent, voice_generator_agent],
    description="이미지와 음성을 병렬로 생성"
)

# 전체 파이프라인
video_pipeline_agent = SequentialAgent(
    name="VideoPipelineAgent",
    sub_agents=[
        script_writer_agent,    # 1. 스크립트 작성
        media_parallel_agent,   # 2. 이미지 + 음성 (병렬)
        render_agent            # 3. 렌더링
    ],
    description="유튜브 영상 생성 파이프라인"
)


# ============================================================
# RUNNER - 파이프라인 실행
# ============================================================

async def run_pipeline(topic: str, duration_minutes: int = 1) -> Optional[str]:
    """ADK 파이프라인 실행"""
    global OUTPUT_DIR, GLOBAL_STATE

    # GLOBAL_STATE 초기화
    GLOBAL_STATE = {
        "script": None,
        "images": None,
        "audio": None,
        "video": None,
    }

    print("=" * 60)
    print("🎬 KRACKER MVP - Google ADK Multi-Agent Pipeline")
    print("=" * 60)
    print(f"📌 주제: {topic}")
    print(f"⏱️  길이: {duration_minutes}분")
    print("=" * 60)

    # 프로젝트 폴더 생성 (주제 기반)
    OUTPUT_DIR = create_project_folder(topic)

    # 세션 서비스 초기화
    session_service = InMemorySessionService()

    # 러너 생성
    runner = Runner(
        agent=video_pipeline_agent,
        app_name="kracker_mvp",
        session_service=session_service
    )

    # 세션 생성
    session = await session_service.create_session(
        app_name="kracker_mvp",
        user_id="user_1"
    )

    # 초기 상태 설정
    session.state["topic"] = topic
    session.state["duration_minutes"] = duration_minutes

    # 파이프라인 실행
    print("\n🚀 파이프라인 시작...")
    print("-" * 40)

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=f"주제: {topic}, 길이: {duration_minutes}분 영상을 만들어주세요.")]
    )

    try:
        async for event in runner.run_async(
            session_id=session.id,
            user_id="user_1",
            new_message=user_message
        ):
            # 이벤트 처리
            if hasattr(event, 'agent_name'):
                print(f"\n📍 Agent: {event.agent_name}")

            if hasattr(event, 'content') and event.content:
                # 결과 출력
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text[:200]
                        print(f"   {text}...")

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("\n⚠️  API 할당량 초과 (429 RESOURCE_EXHAUSTED)")
            print("    → 1-2분 기다린 후 다시 시도하세요.")
            print("    → 또는 다른 API 키를 사용하세요.")
        elif "404" in error_str or "NOT_FOUND" in error_str:
            print(f"\n⚠️  모델을 찾을 수 없음 (404 NOT_FOUND)")
            print(f"    → 모델 이름을 확인하세요: {error_str[:100]}")
        else:
            print(f"\n❌ 파이프라인 에러: {e}")
            import traceback
            traceback.print_exc()
        return None

    # 결과 확인
    print("\n" + "-" * 40)

    # 디버그: 현재 state 출력 (GLOBAL_STATE 사용)
    print("\n📊 State 디버그 (GLOBAL_STATE):")
    print(f"  - script: {'있음' if GLOBAL_STATE.get('script') else '없음'}")
    print(f"  - images: {'있음' if GLOBAL_STATE.get('images') else '없음'}")
    print(f"  - audio: {'있음' if GLOBAL_STATE.get('audio') else '없음'}")
    print(f"  - video: {'있음' if GLOBAL_STATE.get('video') else '없음'}")

    # JSON 출력 저장 (GLOBAL_STATE 사용)
    script_data = GLOBAL_STATE.get("script") or {}
    images_data = GLOBAL_STATE.get("images") or {}
    audio_data = GLOBAL_STATE.get("audio") or {}

    if script_data and (images_data or audio_data):
        print("\n📄 JSON 출력 저장 중...")
        # 스크립트가 문자열이면 JSON으로 파싱
        if isinstance(script_data, str):
            try:
                script_data = json.loads(script_data)
            except:
                script_data = {"title": "Unknown", "scenes": []}
        _save_output_json(script_data, audio_data or {}, images_data or {})

    video_result = GLOBAL_STATE.get("video") or {}
    video_path = video_result.get("video_path", "") if isinstance(video_result, dict) else ""

    # state.video가 없어도 프로젝트 폴더에서 영상 확인
    video_dir = OUTPUT_DIR / "video" if (OUTPUT_DIR / "video").exists() else OUTPUT_DIR
    fallback_video_path = video_dir / "final_video.mp4"

    if video_path and Path(video_path).exists():
        print("\n" + "=" * 60)
        print("✅ 완료!")
        print("=" * 60)
        print(f"📁 프로젝트 폴더: {OUTPUT_DIR}")
        print(f"📁 폴더 구조:")
        print(f"   ├── images/      - 씬별 이미지")
        print(f"   ├── audio/       - 나레이션 음성")
        print(f"   ├── video/       - 최종 영상")
        print(f"   └── metadata/    - JSON 메타데이터")
        print(f"🎥 영상 파일: {video_path}")
        print("=" * 60)
        return video_path
    elif fallback_video_path.exists():
        print("\n" + "=" * 60)
        print("✅ 완료! (fallback)")
        print("=" * 60)
        print(f"📁 프로젝트 폴더: {OUTPUT_DIR}")
        print(f"📁 폴더 구조:")
        print(f"   ├── images/      - 씬별 이미지")
        print(f"   ├── audio/       - 나레이션 음성")
        print(f"   ├── video/       - 최종 영상")
        print(f"   └── metadata/    - JSON 메타데이터")
        print(f"🎥 영상 파일: {fallback_video_path}")
        print("=" * 60)
        return str(fallback_video_path)
    else:
        print("\n❌ 영상 생성 실패")
        print("💡 수동 렌더링 시도...")

        # 수동으로 렌더링 시도 (GLOBAL_STATE 사용, tool_context 없이 직접 호출)
        images = GLOBAL_STATE.get("images") or {}
        audio = GLOBAL_STATE.get("audio") or {}

        if images and audio:
            try:
                image_paths = images.get("image_paths", [])
                audio_path_val = audio.get("audio_path", "")
                timestamps = audio.get("timestamps", [])

                if image_paths and audio_path_val:
                    # _fallback_moviepy_render 직접 호출 (tool_context 불필요)
                    result = _fallback_moviepy_render(image_paths, audio_path_val, timestamps)
                    if result.get("success"):
                        GLOBAL_STATE["video"] = result  # 수동으로 state 업데이트
                        print(f"✅ 수동 렌더링 성공: {result['video_path']}")
                        return result["video_path"]
            except Exception as render_err:
                print(f"❌ 수동 렌더링 실패: {render_err}")

        return None


# ============================================================
# FALLBACK - ADK 없을 때 기본 파이프라인
# ============================================================

def run_simple_pipeline(topic: str, duration_minutes: int = 1) -> Optional[str]:
    """ADK 없을 때 단순 파이프라인"""
    global OUTPUT_DIR

    print("=" * 60)
    print("🎬 KRACKER MVP - Simple Pipeline (No ADK)")
    print("=" * 60)
    print(f"📌 주제: {topic}")
    print(f"⏱️  길이: {duration_minutes}분")
    print("=" * 60)

    # Step 1: 스크립트 생성
    print("\n📝 Step 1: 스크립트 생성 중...")
    num_scenes = max(2, duration_minutes * 60 // 15)

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""당신은 유튜브 교육 콘텐츠 스크립트 작가입니다.

주제: {topic}
영상 길이: {duration_minutes}분
씬 개수: {num_scenes}개

JSON 형식으로 출력:
{{
    "title": "영상 제목",
    "scenes": [
        {{
            "id": 1,
            "narration": "나레이션 (한국어)",
            "visual_description": "Image description (English)",
            "duration_seconds": 15
        }}
    ]
}}

JSON만 출력하세요."""
    )

    text = response.text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        script = json.loads(text.strip())
    except:
        script = {
            "title": topic,
            "scenes": [{"id": 1, "narration": topic, "visual_description": "AI concept art", "duration_seconds": 60}]
        }

    print(f"  ✓ 제목: {script['title']}")
    print(f"  ✓ 씬 개수: {len(script['scenes'])}")

    # 프로젝트 폴더 생성
    OUTPUT_DIR = create_project_folder(script['title'])

    # 스크립트 저장 (metadata 폴더에)
    metadata_dir = OUTPUT_DIR / "metadata"
    with open(metadata_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script, f, ensure_ascii=False, indent=2)

    # Step 2: 이미지 생성
    print("\n🎨 Step 2: 이미지 생성 중...")
    visual_descriptions = [s["visual_description"] for s in script["scenes"]]
    images_result = generate_images_tool(visual_descriptions)
    print(f"  ✓ 이미지: {images_result['count']}장")

    # Step 3: 음성 생성
    print("\n🎙️ Step 3: 음성 생성 중...")
    narrations = [s["narration"] for s in script["scenes"]]
    audio_result = generate_audio_tool(narrations)
    print(f"  ✓ 음성: {audio_result['audio_path']}")

    # Step 3.5: JSON 출력 저장
    print("\n📄 Step 3.5: JSON 출력 저장 중...")
    _save_output_json(script, audio_result, images_result)

    # Step 4: 영상 렌더링
    print("\n🎬 Step 4: 영상 렌더링 중...")
    video_result = render_video_tool(
        images_result["image_paths"],
        audio_result["audio_path"],
        audio_result["timestamps"]
    )

    if video_result["success"]:
        print("\n" + "=" * 60)
        print("✅ 완료!")
        print("=" * 60)
        print(f"📁 프로젝트 폴더: {OUTPUT_DIR}")
        print(f"📁 폴더 구조:")
        print(f"   ├── images/      - 씬별 이미지")
        print(f"   ├── audio/       - 나레이션 음성")
        print(f"   ├── video/       - 최종 영상")
        print(f"   └── metadata/    - JSON 메타데이터")
        print(f"🎥 영상 파일: {video_result['video_path']}")
        print("=" * 60)
        return video_result["video_path"]
    else:
        print("\n❌ 실패")
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("사용법: python mvp_video_pipeline.py \"주제\" [분]")
        print("예시: python mvp_video_pipeline.py \"AI가 바꾸는 미래 직업\" 1")
        return

    topic = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 1

    # ADK 사용 가능 여부 확인
    try:
        from google.adk.agents import LlmAgent
        print("✓ Google ADK 감지됨")
        asyncio.run(run_pipeline(topic, duration))
    except ImportError:
        print("⚠ Google ADK 미설치. 단순 파이프라인 사용")
        run_simple_pipeline(topic, duration)


if __name__ == "__main__":
    main()
