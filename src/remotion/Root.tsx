import React from "react";
import { Composition } from "remotion";
import { VideoComposition } from "./VideoComposition";
import type { PipelineOutput } from "./types";

// 런타임에 public/output/pipeline_output.json에서 데이터 로드
// render-video.js 스크립트가 프로젝트 폴더에서 복사해줌
let pipelineData: PipelineOutput;

try {
  // Remotion 빌드 시 public/ 폴더의 JSON을 로드
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  pipelineData = require("../../public/output/pipeline_output.json");
} catch {
  // 기본 데이터 (Remotion Studio 미리보기용)
  pipelineData = {
    generated_at: new Date().toISOString(),
    title: "KRACKER Video Preview",
    scene_count: 4,
    files: {
      script: "script.json",
      narration: "narration.json",
      images: "images.json",
      audio: "narration.mp3",
      video: "final_video_remotion.mp4",
    },
    script: {
      title: "AI가 바꾸는 미래 직업, 무엇을 준비해야 할까?",
      scenes: [
        {
          id: 1,
          narration: "인공지능이 빠르게 발전하면서 우리의 직업 환경도 크게 변화하고 있습니다. 어떤 직업은 사라지고, 어떤 직업은 새롭게 등장할까요?",
          visual_description: "Fast-paced montage of various professions with AI icons",
          duration_seconds: 15,
        },
        {
          id: 2,
          narration: "단순 반복 업무나 데이터 처리 중심의 직업은 AI로 대체될 가능성이 높습니다. 하지만 창의적인 사고, 공감 능력, 문제 해결 능력이 필요한 직업은 더욱 중요해질 것입니다.",
          visual_description: "Split screen: robot vs human artist",
          duration_seconds: 15,
        },
        {
          id: 3,
          narration: "미래에는 AI를 활용하고 관리하는 능력이 핵심 역량이 될 것입니다. 데이터 분석, AI 윤리, AI 모델 개발 등 새로운 분야에서 기회를 찾을 수 있습니다.",
          visual_description: "People collaborating with AI visualizations",
          duration_seconds: 15,
        },
        {
          id: 4,
          narration: "변화에 대한 열린 마음과 끊임없는 학습 자세를 갖추세요. 미래는 준비하는 자에게 기회를 제공할 것입니다! 구독과 좋아요, 알림 설정 잊지 마세요!",
          visual_description: "Call to action with subscribe button",
          duration_seconds: 15,
        },
      ],
    },
    audio: {
      audio_path: "output/narration.mp3",
      timestamps: [
        { scene_id: 1, start: 0, end: 15 },
        { scene_id: 2, start: 15, end: 30 },
        { scene_id: 3, start: 30, end: 45 },
        { scene_id: 4, start: 45, end: 60 },
      ],
      duration: 60,
    },
    images: {
      count: 4,
      image_paths: [
        "output/scene_1.png",
        "output/scene_2.png",
        "output/scene_3.png",
        "output/scene_4.png",
      ],
      scenes: [
        { scene_id: 1, prompt: "Scene 1 prompt", image_path: "output/scene_1.png" },
        { scene_id: 2, prompt: "Scene 2 prompt", image_path: "output/scene_2.png" },
        { scene_id: 3, prompt: "Scene 3 prompt", image_path: "output/scene_3.png" },
        { scene_id: 4, prompt: "Scene 4 prompt", image_path: "output/scene_4.png" },
      ],
    },
  };
}

// 총 프레임 수 계산 (30fps 기준)
const calculateTotalFrames = (data: PipelineOutput): number => {
  const FPS = 30;
  const duration = data.audio.duration ||
    data.script.scenes.reduce((acc, s) => acc + (s.duration_seconds || 10), 0);
  return Math.ceil(duration * FPS);
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="KrackerVideo"
        component={VideoComposition}
        durationInFrames={calculateTotalFrames(pipelineData)}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{
          pipelineData: pipelineData,
        }}
      />
    </>
  );
};

export default RemotionRoot;
