"""
Remotion 테스트 스크립트
기존 output 파일들을 사용해서 Remotion 렌더링 테스트
"""

import json
from pathlib import Path
import subprocess
import shutil

# 경로 설정
OUTPUT_DIR = Path(__file__).parent / "output"
PROJECT_ROOT = Path(__file__).parent.parent
PUBLIC_OUTPUT = PROJECT_ROOT / "public" / "output"

def get_audio_duration(audio_path: str) -> float:
    """오디오 파일의 길이를 가져옴 (초)"""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(audio_path)
        return audio.info.length
    except ImportError:
        # mutagen이 없으면 기본값 사용
        print("⚠️ mutagen 미설치. 기본 40초로 설정.")
        return 40.0
    except Exception as e:
        print(f"⚠️ 오디오 길이 감지 실패: {e}. 기본 40초로 설정.")
        return 40.0

def create_test_pipeline_data():
    """테스트용 파이프라인 데이터 생성"""

    # 기존 파일 확인
    image_paths = []
    for i in range(1, 5):
        img_path = OUTPUT_DIR / f"scene_{i}.png"
        if img_path.exists():
            image_paths.append(str(img_path))
            print(f"✅ 이미지 발견: {img_path.name}")
        else:
            print(f"❌ 이미지 없음: {img_path.name}")

    audio_path = OUTPUT_DIR / "narration.mp3"
    if audio_path.exists():
        print(f"✅ 오디오 발견: {audio_path.name}")
    else:
        print(f"❌ 오디오 없음: {audio_path.name}")
        return None

    # 오디오 길이 감지
    audio_duration = get_audio_duration(str(audio_path))
    print(f"📊 오디오 길이: {audio_duration:.2f}초")

    # 씬당 시간 계산
    scene_count = len(image_paths)
    scene_duration = audio_duration / scene_count if scene_count > 0 else 10

    # 파이프라인 데이터 생성
    pipeline_data = {
        "generated_at": "2026-02-03T00:00:00.000Z",
        "title": "Remotion 테스트 영상",
        "scene_count": scene_count,
        "files": {
            "script": str(OUTPUT_DIR / "script.json"),
            "narration": str(OUTPUT_DIR / "narration.json"),
            "images": str(OUTPUT_DIR / "images.json"),
            "audio": str(audio_path),
            "video": str(OUTPUT_DIR / "final_video_remotion.mp4")
        },
        "script": {
            "title": "Remotion 테스트 영상",
            "scenes": []
        },
        "audio": {
            "audio_path": str(audio_path),
            "timestamps": [],
            "duration": audio_duration
        },
        "images": {
            "count": scene_count,
            "image_paths": image_paths,
            "scenes": []
        }
    }

    # 씬 정보 생성
    current_time = 0
    for i in range(scene_count):
        # 스크립트 씬
        pipeline_data["script"]["scenes"].append({
            "id": i + 1,
            "narration": f"테스트 씬 {i + 1} 나레이션",
            "visual_description": f"Scene {i + 1} visual description",
            "duration_seconds": scene_duration
        })

        # 타임스탬프
        pipeline_data["audio"]["timestamps"].append({
            "scene_id": i + 1,
            "start": current_time,
            "end": current_time + scene_duration
        })

        # 이미지 씬 정보
        pipeline_data["images"]["scenes"].append({
            "scene_id": i + 1,
            "prompt": f"Scene {i + 1} image prompt",
            "image_path": image_paths[i] if i < len(image_paths) else ""
        })

        current_time += scene_duration

    return pipeline_data

def copy_files_to_public():
    """output 파일들을 public/output으로 복사"""
    PUBLIC_OUTPUT.mkdir(parents=True, exist_ok=True)

    # 이미지 복사
    for i in range(1, 5):
        src = OUTPUT_DIR / f"scene_{i}.png"
        dst = PUBLIC_OUTPUT / f"scene_{i}.png"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"📁 복사: {src.name} → public/output/")

    # 오디오 복사
    audio_src = OUTPUT_DIR / "narration.mp3"
    audio_dst = PUBLIC_OUTPUT / "narration.mp3"
    if audio_src.exists():
        shutil.copy2(audio_src, audio_dst)
        print(f"📁 복사: {audio_src.name} → public/output/")

def run_remotion_render():
    """Remotion 렌더링 실행"""
    print("\n🎬 Remotion 렌더링 시작...")

    try:
        result = subprocess.run(
            ["npm", "run", "remotion:render"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            shell=True
        )

        if result.returncode == 0:
            print("✅ Remotion 렌더링 성공!")
            print(result.stdout)
        else:
            print("❌ Remotion 렌더링 실패:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ 렌더링 오류: {e}")
        return False

    return True

def main():
    print("=" * 60)
    print("🎬 Remotion 테스트 스크립트")
    print("=" * 60)

    # 1. 테스트 데이터 생성
    print("\n📝 Step 1: 테스트 파이프라인 데이터 생성")
    pipeline_data = create_test_pipeline_data()

    if not pipeline_data:
        print("❌ 필요한 파일이 없습니다.")
        return

    # 2. JSON 저장
    print("\n💾 Step 2: JSON 파일 저장")

    # pipeline_output.json
    pipeline_path = OUTPUT_DIR / "pipeline_output.json"
    with open(pipeline_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_data, f, ensure_ascii=False, indent=2)
    print(f"  - {pipeline_path.name}")

    # script.json
    script_path = OUTPUT_DIR / "script.json"
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_data["script"], f, ensure_ascii=False, indent=2)
    print(f"  - {script_path.name}")

    # narration.json
    narration_path = OUTPUT_DIR / "narration.json"
    with open(narration_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_data["audio"], f, ensure_ascii=False, indent=2)
    print(f"  - {narration_path.name}")

    # images.json
    images_path = OUTPUT_DIR / "images.json"
    with open(images_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_data["images"], f, ensure_ascii=False, indent=2)
    print(f"  - {images_path.name}")

    # 3. public/output에 파일 복사
    print("\n📁 Step 3: public/output에 파일 복사")
    copy_files_to_public()

    # 4. Remotion 렌더링
    print("\n🎬 Step 4: Remotion 렌더링")
    success = run_remotion_render()

    # 결과
    print("\n" + "=" * 60)
    if success:
        output_video = OUTPUT_DIR / "final_video_remotion.mp4"
        if output_video.exists():
            print(f"✅ 완료! 영상 생성됨: {output_video}")
        else:
            print("⚠️ 렌더링은 성공했으나 영상 파일을 찾을 수 없습니다.")
    else:
        print("❌ 렌더링 실패. 로그를 확인하세요.")
    print("=" * 60)

if __name__ == "__main__":
    main()
