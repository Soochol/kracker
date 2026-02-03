import React from "react";
import { Img, interpolate, Easing, useCurrentFrame } from "remotion";
import type { KenBurnsEffect, KenBurnsSceneProps } from "../types";

// Ken Burns 효과별 설정
const EFFECTS_CONFIG: Record<
  KenBurnsEffect,
  {
    scaleStart: number;
    scaleEnd: number;
    panX: [number, number];
    panY: [number, number];
  }
> = {
  zoomIn: {
    scaleStart: 1.0,
    scaleEnd: 1.15,
    panX: [0, 30],
    panY: [0, -15],
  },
  zoomOut: {
    scaleStart: 1.15,
    scaleEnd: 1.0,
    panX: [30, 0],
    panY: [-15, 0],
  },
  panLeft: {
    scaleStart: 1.08,
    scaleEnd: 1.08,
    panX: [60, -60],
    panY: [0, 0],
  },
  panRight: {
    scaleStart: 1.08,
    scaleEnd: 1.08,
    panX: [-60, 60],
    panY: [0, 0],
  },
  slowPush: {
    scaleStart: 1.0,
    scaleEnd: 1.08,
    panX: [0, 0],
    panY: [0, -10],
  },
};

export const KenBurnsScene: React.FC<KenBurnsSceneProps> = ({
  src,
  durationInFrames,
  effect,
}) => {
  const frame = useCurrentFrame();
  const progress = frame / durationInFrames;

  const config = EFFECTS_CONFIG[effect];
  const { scaleStart, scaleEnd, panX, panY } = config;

  // 부드러운 이징 적용
  const scale = interpolate(progress, [0, 1], [scaleStart, scaleEnd], {
    easing: Easing.inOut(Easing.cubic),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const translateX = interpolate(progress, [0, 1], panX, {
    easing: Easing.inOut(Easing.cubic),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const translateY = interpolate(progress, [0, 1], panY, {
    easing: Easing.inOut(Easing.cubic),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        backgroundColor: "#000",
      }}
    >
      <Img
        src={src}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: `scale(${scale}) translate(${translateX}px, ${translateY}px)`,
          transformOrigin: "center center",
        }}
      />
    </div>
  );
};

export default KenBurnsScene;
