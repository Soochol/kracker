// Remotion 타입 정의

export type KenBurnsEffect = "zoomIn" | "zoomOut" | "panLeft" | "panRight" | "slowPush";

export interface Scene {
  id: number;
  narration: string;
  visual_description: string;
  duration_seconds: number;
}

export interface Script {
  title: string;
  scenes: Scene[];
}

export interface Timestamp {
  scene_id: number;
  start: number;
  end: number;
}

export interface AudioResult {
  audio_path: string;
  timestamps: Timestamp[];
  duration: number;
}

export interface ImagesResult {
  count: number;
  image_paths: string[];
  scenes: {
    id: number;
    visual_description: string;
    image_path: string;
  }[];
}

export interface PipelineOutput {
  generated_at: string;
  title: string;
  scene_count: number;
  files: {
    script: string;
    narration: string;
    images: string;
    audio: string;
    video: string;
  };
  script: Script;
  audio: AudioResult;
  images: ImagesResult;
}

export interface VideoCompositionProps {
  pipelineData: PipelineOutput;
}

export interface KenBurnsSceneProps {
  src: string;
  durationInFrames: number;
  effect: KenBurnsEffect;
}
