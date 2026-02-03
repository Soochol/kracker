import { Config } from "@remotion/cli/config";

// 비디오 이미지 형식
Config.setVideoImageFormat("png");

// 출력 위치
Config.setOutputLocation("./test-ltx/output/final_video_remotion.mp4");

// 브라우저 타임아웃 (복잡한 렌더링용)
Config.setChromiumOpenGlRenderer("angle");

// 병렬 처리
Config.setConcurrency(4);
