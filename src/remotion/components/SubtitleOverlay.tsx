import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  Easing,
} from "remotion";

export interface SubtitleProps {
  text: string;
  durationInFrames: number;
  style?: "default" | "minimal" | "cinematic";
}

export const SubtitleOverlay: React.FC<SubtitleProps> = ({
  text,
  durationInFrames,
  style = "default",
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 페이드 인/아웃 애니메이션 (0.3초)
  const fadeFrames = Math.round(fps * 0.3);

  const opacity = interpolate(
    frame,
    [0, fadeFrames, durationInFrames - fadeFrames, durationInFrames],
    [0, 1, 1, 0],
    {
      easing: Easing.inOut(Easing.cubic),
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );

  // 약간의 위로 올라오는 애니메이션
  const translateY = interpolate(
    frame,
    [0, fadeFrames],
    [10, 0],
    {
      easing: Easing.out(Easing.cubic),
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );

  // 스타일별 설정
  const styles = {
    default: {
      container: {
        position: "absolute" as const,
        bottom: 80,
        left: 0,
        right: 0,
        display: "flex",
        justifyContent: "center",
        padding: "0 60px",
      },
      text: {
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        color: "#ffffff",
        fontSize: 42,
        fontFamily: "'Pretendard', 'Noto Sans KR', sans-serif",
        fontWeight: 500,
        padding: "16px 32px",
        borderRadius: 8,
        textAlign: "center" as const,
        maxWidth: "80%",
        lineHeight: 1.5,
        textShadow: "0 2px 4px rgba(0,0,0,0.5)",
      },
    },
    minimal: {
      container: {
        position: "absolute" as const,
        bottom: 60,
        left: 0,
        right: 0,
        display: "flex",
        justifyContent: "center",
        padding: "0 40px",
      },
      text: {
        color: "#ffffff",
        fontSize: 38,
        fontFamily: "'Pretendard', 'Noto Sans KR', sans-serif",
        fontWeight: 400,
        textAlign: "center" as const,
        maxWidth: "85%",
        lineHeight: 1.6,
        textShadow: "0 2px 8px rgba(0,0,0,0.9), 0 0 20px rgba(0,0,0,0.5)",
      },
    },
    cinematic: {
      container: {
        position: "absolute" as const,
        bottom: 100,
        left: 0,
        right: 0,
        display: "flex",
        justifyContent: "center",
        padding: "0 80px",
      },
      text: {
        backgroundColor: "rgba(0, 0, 0, 0.6)",
        color: "#ffffff",
        fontSize: 36,
        fontFamily: "'Pretendard', 'Noto Sans KR', serif",
        fontWeight: 300,
        padding: "20px 40px",
        borderRadius: 4,
        textAlign: "center" as const,
        maxWidth: "75%",
        lineHeight: 1.7,
        letterSpacing: "0.5px",
        borderLeft: "3px solid rgba(255,255,255,0.3)",
        borderRight: "3px solid rgba(255,255,255,0.3)",
      },
    },
  };

  const currentStyle = styles[style];

  return (
    <div
      style={{
        ...currentStyle.container,
        opacity,
        transform: `translateY(${translateY}px)`,
      }}
    >
      <div style={currentStyle.text}>{text}</div>
    </div>
  );
};

export default SubtitleOverlay;
