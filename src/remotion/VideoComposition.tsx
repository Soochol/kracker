import React from "react";
import { Sequence, Audio, AbsoluteFill, staticFile } from "remotion";
import { KenBurnsScene } from "./components/KenBurnsScene";
import { SubtitleOverlay } from "./components/SubtitleOverlay";
import type { KenBurnsEffect, VideoCompositionProps } from "./types";

// Ken Burns 효과 순환 패턴
const KEN_BURNS_PATTERN: KenBurnsEffect[] = [
  "zoomIn",
  "slowPush",
  "panRight",
  "zoomOut",
  "panLeft",
];

export const VideoComposition: React.FC<VideoCompositionProps> = ({
  pipelineData,
}) => {
  const FPS = 30;
  const { script, images, audio } = pipelineData;

  // 씬별 프레임 계산
  let currentFrame = 0;
  const sceneFrames = script.scenes.map((scene, idx) => {
    // 타임스탬프가 있으면 사용, 없으면 duration_seconds 사용
    const timestamp = audio.timestamps?.[idx];
    let durationSeconds: number;

    if (timestamp) {
      durationSeconds = timestamp.end - timestamp.start;
    } else {
      durationSeconds = scene.duration_seconds || audio.duration / script.scenes.length;
    }

    const durationInFrames = Math.max(Math.round(durationSeconds * FPS), FPS); // 최소 1초
    const fromFrame = currentFrame;
    currentFrame += durationInFrames;

    return {
      scene,
      idx,
      fromFrame,
      durationInFrames,
      imagePath: images.image_paths[idx] || "",
      effect: KEN_BURNS_PATTERN[idx % KEN_BURNS_PATTERN.length],
    };
  });

  // 이미지 경로를 Remotion이 접근할 수 있는 형태로 변환
  const getImageSrc = (imagePath: string) => {
    // Windows 경로를 URL로 변환
    const fileName = imagePath.split(/[/\\]/).pop() || "";
    // public 폴더의 output 디렉토리 참조
    return staticFile(`output/${fileName}`);
  };

  const getAudioSrc = () => {
    const fileName = audio.audio_path.split(/[/\\]/).pop() || "";
    return staticFile(`output/${fileName}`);
  };

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {/* 씬별 이미지 + Ken Burns 효과 */}
      {sceneFrames.map(({ idx, fromFrame, durationInFrames, imagePath, effect }) => (
        <Sequence
          key={`image-${idx}`}
          from={fromFrame}
          durationInFrames={durationInFrames}
          name={`Scene ${idx + 1}`}
        >
          <KenBurnsScene
            src={getImageSrc(imagePath)}
            durationInFrames={durationInFrames}
            effect={effect}
          />
        </Sequence>
      ))}

      {/* 자막 레이어 */}
      {sceneFrames.map(({ scene, idx, fromFrame, durationInFrames }) => (
        <Sequence
          key={`subtitle-${idx}`}
          from={fromFrame}
          durationInFrames={durationInFrames}
          name={`Subtitle ${idx + 1}`}
        >
          <SubtitleOverlay
            text={scene.narration}
            durationInFrames={durationInFrames}
            style="default"
          />
        </Sequence>
      ))}

      {/* 오디오 트랙 */}
      {audio.audio_path && <Audio src={getAudioSrc()} />}
    </AbsoluteFill>
  );
};

export default VideoComposition;
