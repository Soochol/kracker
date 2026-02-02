import torch
import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env 파일에서 환경변수 로드
load_dotenv()

# Gemini API 설정
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY가 .env 파일에 없습니다.")
else:
    genai.configure(api_key=api_key)

# ===== GPU 확인 =====
if not torch.cuda.is_available():
    print("ERROR: GPU를 찾을 수 없습니다.")
    print("ROCm + PyTorch 설치 필요:")
    print("  pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio")
    exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")


# ===== Gemini API로 프롬프트 생성 =====
def generate_prompt_with_gemini(topic: str = None) -> str:
    """Gemini API를 사용해 비디오 생성용 프롬프트를 만듭니다."""

    if topic:
        gemini_prompt = f"""Create a single, detailed video generation prompt about: {topic}

Requirements:
- One sentence only, max 50 words
- Include: subject, action, setting, mood, visual style
- Example format: "A [subject] [action] in [setting], [mood], [visual style]"
- Output ONLY the prompt, nothing else"""
    else:
        gemini_prompt = """Create a single, creative and visually interesting video generation prompt.

Requirements:
- One sentence only, max 50 words
- Include: subject, action, setting, mood, visual style
- Example format: "A [subject] [action] in [setting], [mood], [visual style]"
- Be creative and cinematic
- Output ONLY the prompt, nothing else"""

    try:
        print("Gemini API로 프롬프트 생성 중...")
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(gemini_prompt)

        prompt = response.text.strip()
        # 빈 줄이나 불필요한 텍스트 제거
        lines = [line.strip() for line in prompt.split('\n') if line.strip()]
        prompt = lines[-1] if lines else prompt

        # 따옴표 제거
        prompt = prompt.strip('"').strip("'")

        return prompt

    except Exception as e:
        print(f"Gemini API 에러: {e}")
        return None


# ===== 설정 =====
USE_GEMINI = True   # Gemini로 프롬프트 생성
TOPIC = None        # 특정 주제 (None이면 랜덤, 예: "우주 고양이")
UPSCALE = True      # 업스케일 활성화 여부
USE_FP16 = False    # AMD GPU는 fp32 권장

# 기본 프롬프트 (Gemini 실패 시 사용)
DEFAULT_PROMPT = "A cat walking slowly in the snow, cinematic, 4K quality"

dtype = torch.float16 if USE_FP16 else torch.float32

# ===== 프롬프트 생성 =====
if USE_GEMINI and api_key:
    prompt = generate_prompt_with_gemini(TOPIC)
    if not prompt:
        print(f"기본 프롬프트 사용: {DEFAULT_PROMPT}")
        prompt = DEFAULT_PROMPT
else:
    prompt = DEFAULT_PROMPT

print(f"\n프롬프트: {prompt}\n")

# ===== 1단계: 저해상도 빠른 생성 =====
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

print("모델 로딩 중... (첫 실행 시 다운로드)")
pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=dtype
)
pipe.to("cuda")

print("1단계: 480p 비디오 생성 중... (저해상도 = 빠른 생성)")

output = pipe(
    prompt=prompt,
    num_frames=25,           # 8n+1 형식
    width=640,               # 480p 해상도
    height=480,
    num_inference_steps=20,  # 낮을수록 빠름
    output_type="latent" if UPSCALE else "pil",
)

if not UPSCALE:
    export_to_video(output.frames[0], "output_480p.mp4", fps=8)
    print("저장 완료: output_480p.mp4")
else:
    print("1단계 완료 (latent 유지)")

# ===== 2단계: 업스케일링 =====
if UPSCALE:
    print("2단계: 업스케일링 중...")

    try:
        from diffusers import LTXLatentUpsamplePipeline

        upscaler = LTXLatentUpsamplePipeline.from_pretrained(
            "Lightricks/ltxv-spatial-upscaler-0.9.7",
            vae=pipe.vae,
            torch_dtype=dtype
        )
        upscaler.to("cuda")

        upscaled = upscaler(
            latents=output.latents,
            output_type="pil"
        )

        export_to_video(upscaled.frames[0], "output_upscaled.mp4", fps=8)
        print("저장 완료: output_upscaled.mp4 (업스케일됨)")

    except ImportError:
        print("WARNING: LTXLatentUpsamplePipeline을 찾을 수 없습니다.")
        print("diffusers dev 브랜치 설치 필요")
        frames = pipe.vae.decode(output.latents).sample
        export_to_video(frames, "output_480p.mp4", fps=8)
        print("저장 완료: output_480p.mp4 (원본)")

    except Exception as e:
        print(f"업스케일 에러: {e}")

print("\n완료!")
